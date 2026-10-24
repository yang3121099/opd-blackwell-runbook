"""Q-062 difficulty probe (runbook [v2.5.9z74]; Runming 10-22 16:0x Beijing "how to tell difficulty without a label", pre-registered in
research_program 10-22): can a problem's difficulty be read off the 1.7B base before it answers?

1. One forward pass of the base over each prompt rendered with the student's template (non-thinking, as diff_labels / the trainer):
   the last prompt token's hidden state at every layer (the embeddings and each block), and the entropy of the next-token
   distribution there (proxy 1, "first-token entropy"; score = -entropy, lower = easier).
2. Labels: PROBE-DIFF17's labels_17.jsonl, easy = r >= 0.5 (the 1.7B base non-thinking, k = 8): the 6,400 train prompts
   (diff_labels.train_prompts, matched by key) and MATH-500's 500 (by position).
3. Per layer: features standardized on train, L2 logistic regression; the strength from L2_GRID by 5-fold CV AUC on train; the
   5-fold CV AUC on train and the AUC on MATH-500 of the fit on all of train. H1's layer = the best by train CV AUC (MATH-500 never
   picks it); the best MATH-500 layer is reported too, as a description.
4. Proxy 2 (MATH-500): a 256-token greedy attempt on the non-thinking template, then one more user turn asking for a confidence
   digit 0-9; score = the expected digit under the next-token distribution restricted to "0".."9".
5. H3: a MATH-500 16K reading of the route arm (gen_eval jsonl: pid, n_tokens): mean length hard / easy with the true labels and
   with the probe's (p_easy >= 0.5 at H1's layer); H3 holds when the two ratios differ by < 0.2.
Pre-registered lines: H1 AUC >= 0.8 at some layer (H1's layer); H2 probe AUC > first-token-entropy AUC; H3 as above.
The numerics (auc, the logistic fit, cv, the ratio) are plain numpy and run without a GPU (test_diff_probe.py); torch / the model
only in main(). Usage: python diff_probe.py --model <base> --labels <labels_17.jsonl> --train <dapo_train.parquet> --out <dir>
[--route <reading.jsonl>] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CONF_Q = ("Before you go on: on a scale from 0 (certainly wrong) to 9 (certainly right), how confident are you that your final "
          "answer to this problem will be correct? Reply with a single digit.")


def auc(y, s):
    """ROC AUC of scores s for labels y (1 = positive), ties counted half (Mann-Whitney); None without both classes."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logreg(X, y, l2, iters=50):
    """L2 logistic regression (mean log-loss + l2 / 2 |w|^2, no penalty on the bias) by Newton steps; (w, b)."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1)
    reg = np.full(d + 1, l2); reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n + reg * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb / n + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1], w[-1]


def torch_fit(torch, dev="cuda"):
    """fit_logreg on the GPU (float32 products, the Newton system solved in float64): one layer's 26 fits take seconds there."""
    def fit(X, y, l2, iters=50):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev); y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev)
        n, d = X.shape; Xb = torch.cat([X, torch.ones(n, 1, device=dev)], 1)
        w = torch.zeros(d + 1, dtype=torch.float64, device=dev)
        reg = torch.full((d + 1,), float(l2), dtype=torch.float64, device=dev); reg[-1] = 0.0
        for _ in range(iters):
            p = torch.sigmoid((Xb @ w.float()).clamp(-30, 30))
            g = (Xb.T @ (p - y)).double() / n + reg * w
            H = ((Xb * (p * (1 - p))[:, None]).T @ Xb).double() / n + torch.diag(reg + 1e-9)
            step = torch.linalg.solve(H, g); w = w - step
            if float(step.abs().max()) < 1e-7:
                break
        w = w.cpu().numpy()
        return w[:-1], w[-1]
    return fit


def predict(w, b, X):
    return 1 / (1 + np.exp(-np.clip(np.asarray(X, dtype=np.float64) @ w + b, -30, 30)))


def folds(n, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(Xtr - mu) / sd] + [(x - mu) / sd for x in others]


def layer_probe(Xtr, ytr, Xte, yte, fit=fit_logreg, grid=L2_GRID, k=5):
    """One layer: {l2, cv_auc, cv_auc_sd, test_auc, p_test}; the strength by k-fold CV AUC on train."""
    fs = folds(len(ytr), k); best = None
    for l2 in grid:
        aucs = []
        for f in fs:
            tr = np.setdiff1d(np.arange(len(ytr)), f)
            a, b = standardize(Xtr[tr], Xtr[f])
            w, c = fit(a, ytr[tr], l2)
            aucs.append(auc(ytr[f], predict(w, c, b)))
        m = float(np.mean(aucs))
        if best is None or m > best[1]:
            best = (l2, m, float(np.std(aucs)))
    a, b = standardize(Xtr, Xte)
    w, c = fit(a, ytr, best[0]); p = predict(w, c, b)
    return {"l2": best[0], "cv_auc": round(best[1], 4), "cv_auc_sd": round(best[2], 4), "test_auc": round(auc(yte, p), 4), "p_test": p}


def length_ratio(recs, easy):
    """mean n_tokens on hard / on easy problems; recs: [{pid index, n_tokens}], easy: {index: bool}; None if a side is empty."""
    h = [r["n_tokens"] for r in recs if easy.get(r["i"]) is False]
    e = [r["n_tokens"] for r in recs if easy.get(r["i"]) is True]
    return round(float(np.mean(h)) / float(np.mean(e)), 4) if h and e else None


def read_route(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append({"i": int(str(r["pid"]).rsplit("-", 1)[-1]), "n_tokens": int(r["n_tokens"])})
    return out


def judge(layers, ent_auc, conf_auc, h3):
    """The pre-registered lines from the per-layer results (H1's layer = best train CV)."""
    h1l = max(range(len(layers)), key=lambda i: layers[i]["cv_auc"])
    tl = max(range(len(layers)), key=lambda i: layers[i]["test_auc"])
    a = layers[h1l]["test_auc"]
    return {"h1_layer": h1l, "h1_auc": a, "H1": bool(a >= 0.8), "best_test_layer": tl, "best_test_auc": layers[tl]["test_auc"],
            "entropy_auc": ent_auc, "confidence_auc": conf_auc, "H2": bool(ent_auc is not None and a > ent_auc),
            "H3": (None if h3.get("ratio_true") is None or h3.get("ratio_probe") is None
                   else bool(abs(h3["ratio_true"] - h3["ratio_probe"]) < 0.2))}


# ---------------------------------------------------------------------------------------------------------------- the model
def forward_feats(model, tok, texts, bs, torch):
    """[N, layers, H] last-token hidden states (float32) and [N] next-token entropies."""
    tok.padding_side = "left"
    hs, ent = [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        hs.append(torch.stack([h[:, -1, :].float() for h in o.hidden_states], 1).cpu().numpy())
        lp = torch.log_softmax(o.logits[:, -1, :].float(), -1)
        ent.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
    return np.concatenate(hs), np.concatenate(ent)


def confidence(model, tok, msgs, bs, torch, max_new=256):
    """[N] expected confidence digit / 9 after a greedy attempt of max_new tokens (non-thinking template)."""
    tok.padding_side = "left"
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    ck = {"enable_thinking": False}
    out = []
    for i in range(0, len(msgs), bs):
        part = msgs[i:i + bs]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **ck) for m in part]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        atts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        q = [tok.apply_chat_template(m + [{"role": "assistant", "content": a}, {"role": "user", "content": CONF_Q}], add_generation_prompt=True,
                                     tokenize=False, **ck) for m, a in zip(part, atts)]
        enc = tok(q, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        pd_ = torch.softmax(lg[:, digits], -1).cpu().numpy()
        out.append((pd_ * np.arange(10)).sum(1) / 9.0)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--route", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--tbs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--maxp", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, HERE); import diff_labels as DL
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    lab = [json.loads(x) for x in open(a.labels) if x.strip()]
    tr_lab = {r["key"]: float(r["r"]) for r in lab if r.get("split") == "train"}
    mp_lab = {int(r["pos"]): float(r["r"]) for r in lab if r.get("split") == "math500"}
    tok = AutoTokenizer.from_pretrained(a.model); ck = {"enable_thinking": False}
    tr, _, _ = DL.train_prompts(a.train, tok, a.steps, a.tbs, a.seed, a.maxp, ck)
    tr = [(m, tr_lab[DL.key(m)]) for _, m, _, _ in tr if DL.key(m) in tr_lab]
    mp = [(m, mp_lab[i]) for i, m, _, _ in DL.math500_prompts() if i in mp_lab]
    if a.limit:
        tr, mp = tr[:a.limit], mp[:max(50, a.limit // 4)]
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    txt = lambda rows: [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **ck) for m, _ in rows]
    Xtr, etr = forward_feats(model, tok, txt(tr), a.bs, torch); Xte, ete = forward_feats(model, tok, txt(mp), a.bs, torch)
    ytr = np.array([r >= 0.5 for _, r in tr], dtype=float); yte = np.array([r >= 0.5 for _, r in mp], dtype=float)
    t1 = time.time()
    conf = confidence(model, tok, [m for m, _ in mp], a.bs, torch)
    t2 = time.time()
    del model; torch.cuda.empty_cache()
    layers = []
    for li in range(Xtr.shape[1]):
        r = layer_probe(Xtr[:, li, :], ytr, Xte[:, li, :], yte, fit=torch_fit(torch))
        layers.append(r)
        print(f"DIFF_PROBE layer {li} cv {r['cv_auc']} test {r['test_auc']} l2 {r['l2']}", flush=True)
    t3 = time.time()
    ent_auc = auc(yte, -ete); conf_auc = auc(yte, conf)
    h1l = max(range(len(layers)), key=lambda i: layers[i]["cv_auc"])
    h3 = {"reading": os.path.basename(a.route) if a.route else None}
    if a.route and os.path.exists(a.route):
        recs = read_route(a.route)
        true = {i: r >= 0.5 for i, r in mp_lab.items()}
        pidx = [i for i, m, _, _ in DL.math500_prompts() if i in mp_lab][:len(mp)]
        probe = {i: bool(p >= 0.5) for i, p in zip(pidx, layers[h1l]["p_test"])}
        h3.update(ratio_true=length_ratio(recs, true), ratio_probe=length_ratio(recs, probe),
                  probe_easy_share=round(float(np.mean(list(probe.values()))), 4), true_easy_share=round(float(np.mean(list(true.values()))), 4))
    rep = {"model": os.path.basename(a.model.rstrip("/")), "n_train": len(tr), "n_math500": len(mp),
           "train_easy_share": round(float(ytr.mean()), 4), "math500_easy_share": round(float(yte.mean()), 4),
           "layers": [{k: v for k, v in r.items() if k != "p_test"} for r in layers], "h3": h3,
           "entropy_auc_train": auc(ytr, -etr), **judge(layers, ent_auc, conf_auc, h3),
           "seconds": {"forward": round(t1 - t0, 1), "confidence": round(t2 - t1, 1), "probes": round(t3 - t2, 1)},
           "lines": {"H1": "AUC >= 0.8 at H1's layer (best train CV)", "H2": "probe AUC > first-token-entropy AUC", "H3": "|ratio_probe - ratio_true| < 0.2"}}
    json.dump(rep, open(os.path.join(a.out, "diff_probe.json"), "w"), indent=1)
    print("DIFF_PROBE", json.dumps({k: rep[k] for k in ("h1_layer", "h1_auc", "H1", "entropy_auc", "confidence_auc", "H2", "H3")}))


if __name__ == "__main__":
    main()
