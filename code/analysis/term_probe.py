"""PROBE-TERM: termination quality of trained students, no training, forward passes only (runbook v2.5.7g; reviewer
2026-10-18 19:0x UTC). Input: each model's own MATH500 T0.6 generations (gen_eval jsonl, not sampled again).
Per generation, with the response tokens as generated (token_ids):
  anchor  = the first complete \\boxed{...} (balanced braces) in the response; generations without one are counted, not
            measured. nl = the first token after the anchor whose text holds a newline, or the end-of-sequence token if
            the generation ends before any newline.
  (1) p_stop_nl: the model's own p(<|im_end|>) at nl (the distribution that chose that token: stop here, or go on), in
      its generation template (non-thinking); also p_stop_after_box, the same at the first token after the anchor.
  (2) marker_frac: of the continuation after the anchor (its tokens, end-of-sequence excluded), the share of tokens
      inside a re-check phrase: wait / let me check / let me verify / double-check / verify / re-check / alternatively /
      however / hmm / actually / on second thought / but let me (case-insensitive, word bounded; MARKERS below; reviewer
      A1 19:2x UTC added the bare verify / re-check); per generation with a
      continuation, and pooled (all marker tokens / all continuation tokens).
  (3) teacher_stop_nl: the 8B teacher's p(<|im_end|>) at the same nl after the same prefix, in the thinking template
      (P5's teacher prompt, OPD_REF_THINK=1); teacher_nothink_stop_nl in the non-thinking template (P2's).
  v2.5.7k (PROBE-TERM2; reviewer 23:35 UTC): nl lands inside the display math that holds the box in most answers
  ("$$\n\boxed{x}\n$$": 96% of E11b-P0's samples), where no model stops. env_nl = the first token after the math
  environment that encloses the box closes ($$ / \\] / \\) / $), whose text holds a newline (or the end-of-sequence token
  if the answer ends first): p_stop_env_nl, teacher_stop_env_nl, teacher_nothink_stop_env_nl; and p_stop_max16 /
  teacher_stop_max16 = the largest p(<|im_end|>) over the 16 tokens after the token that closes the box. The nl measures
  are kept for comparison; ctx_env in samples.jsonl shows 12 bytes either side of the environment's close.
  v2.5.7z (PROBE-TERM4; reviewer 11:1x UTC, closing TERM2 / TERM3): does P5 stop once its thought is closed? With
  --term4, per generation also: k_close = the first </think> the model wrote (token 151668; the non-thinking template's
  own empty think block is in the prompt, not here); the first box holding the final answer (the last box's content);
  p_close_think_max16 = the largest p(</think>) over the 16 tokens after that box's closing brace (box_before_close
  marks it inside the thought); the last box (the summary's) and p_stop_last_env_nl = p(<|im_end|>) at the first newline
  after its math environment closes (or the end token), p_stop_last_max16 = the largest p(<|im_end|>) over the 16 tokens
  after it. A fifth tsv field `think` renders the model's prompt in the thinking template (the base in thinking mode).
Per model: generations, truncation rate, anchor rate, mean / median / 10 25 75 90th percentiles of every measure and a
95% CI of the mean (bootstrap over problems, 10000 resamples, seed 0); within each size group (tag prefix before the
first "_"), every pair of models on (1), (2), (3) (problem-level paired bootstrap of the per-problem means).
Usage: python term_probe.py --teacher <8B dir> --inputs <tsv: tag, model dir, generations jsonl, model id, source>
           --out <report.json> [--samples <per-sample.jsonl>] [--max_per_model N] [--no_teacher_nothink] [--term4]
"""
import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

MARKERS = rb"(?i)\b(?:wait|let me (?:check|verify|double[- ]?check|re-?check|re-?examine|reconsider|confirm|re-?evaluate|make sure)" \
          rb"|double[- ]?check|verify|re-?check|alternatively|however|hmm+|actually|on second thought|but let me)\b"
EOS_IDS = (151645, 151643)   # <|im_end|>, <|endoftext|> (Qwen3); p_stop = p(<|im_end|>), p_stop_any = both
THINK_CLOSE = 151668          # </think> (Qwen3, an added token: its byte piece is empty here)


def bytes_to_unicode():
    """The byte-level BPE alphabet (GPT-2 / Qwen): byte -> printable character."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for x in range(256):
        if x not in bs:
            bs.append(x); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


U2B = {v: k for k, v in bytes_to_unicode().items()}


def token_bytes(tok, ids):
    """The text bytes of each response token (added / special tokens: empty)."""
    u2b = U2B
    special = set(getattr(tok, "added_tokens_decoder", {}) or {}) | set(tok.all_special_ids)
    out = []
    for i, t in zip(ids, tok.convert_ids_to_tokens(ids)):
        if i in special or t is None:
            out.append(b"")
        else:
            out.append(bytes(u2b[c] for c in t) if all(c in u2b for c in t) else t.encode())
    return out


def first_boxed_span(b):
    """(start, end) bytes of the first complete \\boxed{...} (end just past its closing brace), or None."""
    i = b.find(b"\\boxed{")
    while i >= 0:
        depth, j = 1, i + 7
        while j < len(b) and depth:
            depth += {123: 1, 125: -1}.get(b[j], 0); j += 1
        if depth == 0:
            return i, j
        i = b.find(b"\\boxed{", i + 1)
    return None


def first_boxed_end(b):
    """Byte index just past the closing brace of the first complete \\boxed{...}, or None."""
    sp = first_boxed_span(b)
    return sp[1] if sp else None


def env_close(b, s, e):
    """Byte index just past the delimiter that closes the math environment open at s ($$, \\[ ... \\], \\( ... \\), $),
    or e when the box is not inside one (or its environment never closes)."""
    i, st = 0, None
    while i < s:
        if st is None:
            if b.startswith(b"$$", i):
                st = b"$$"; i += 2; continue
            if b.startswith(b"\\[", i):
                st = b"\\]"; i += 2; continue
            if b.startswith(b"\\(", i):
                st = b"\\)"; i += 2; continue
            if b[i:i + 1] == b"$":
                st = b"$"; i += 1; continue
        elif st == b"$" and b[i:i + 1] == b"$" and not b.startswith(b"$$", i):
            st = None; i += 1; continue
        elif st != b"$" and b.startswith(st, i):
            i += len(st); st = None; continue
        i += 1
    if st is None:
        return e
    if st == b"$":
        j = e
        while j < len(b):
            if b[j:j + 1] == b"$" and not b.startswith(b"$$", j):
                return j + 1
            j += 2 if b.startswith(b"$$", j) else 1
        return e
    j = b.find(st, e)
    return j + len(st) if j >= 0 else e


def env_positions(pieces, ids):
    """(k_box, k_env, case_env, ctx): the token holding the box's closing brace, the first token after the enclosing
    math environment's close that holds a newline (env_nl) or the end-of-sequence token if that comes first (env_eos),
    else None (env_none); ctx = 12 bytes either side of the close. None without a complete box."""
    b = b"".join(pieces)
    sp = first_boxed_span(b)
    if sp is None:
        return None
    starts = starts_of(pieces)
    tok_of = lambda byte: bisect.bisect_right(starts, byte) - 1
    k_box = tok_of(sp[1] - 1)
    ec = env_close(b, sp[0], sp[1])
    ctx = b[max(0, ec - 12): ec + 12].decode("utf-8", "replace")
    nl = b.find(b"\n", ec)
    eos = next((k for k in range(k_box + 1, len(ids)) if ids[k] in EOS_IDS and (starts[k] >= ec or k == len(ids) - 1)), None)
    if nl >= 0:
        k = tok_of(nl)
        return (k_box, eos, "env_eos", ctx) if eos is not None and eos < k else (k_box, k, "env_nl", ctx)
    return (k_box, eos, "env_eos", ctx) if eos is not None else (k_box, None, "env_none", ctx)


def box_spans(b):
    """[(start, end, content)] of the complete \\boxed{...} in b (bytes)."""
    out, i = [], b.find(b"\\boxed{")
    while i >= 0:
        depth, j = 1, i + 7
        while j < len(b) and depth:
            depth += {123: 1, 125: -1}.get(b[j], 0); j += 1
        if depth:
            break
        out.append((i, j, b[i + 7: j - 1]))
        i = b.find(b"\\boxed{", j)
    return out


def term4_positions(pieces, ids):
    """v2.5.7z: dict(k_close, k_fb, box_before_close, k_last, k_last_env, case_last) or None without a box: k_close = the
    first </think> token, k_fb = the token holding the closing brace of the first box with the final (last box's)
    content, k_last = the same for the last box, k_last_env = the first token after the last box's math environment
    closes that holds a newline, or the end token if that comes first."""
    b = b"".join(pieces)
    sp = box_spans(b)
    if not sp:
        return None
    starts = starts_of(pieces)
    tok_of = lambda byte: bisect.bisect_right(starts, byte) - 1
    final = b"".join(sp[-1][2].split())
    fb = next(s for s in sp if b"".join(s[2].split()) == final)
    k_close = next((k for k, t in enumerate(ids) if t == THINK_CLOSE), None)
    k_fb, k_last = tok_of(fb[1] - 1), tok_of(sp[-1][1] - 1)
    ec = env_close(b, sp[-1][0], sp[-1][1])
    nl = b.find(b"\n", ec)
    eos = next((k for k in range(k_last + 1, len(ids)) if ids[k] in EOS_IDS and (starts[k] >= ec or k == len(ids) - 1)), None)
    if nl >= 0 and (eos is None or tok_of(nl) < eos):
        k_env, case = tok_of(nl), "env_nl"
    else:
        k_env, case = eos, ("env_eos" if eos is not None else "env_none")
    return {"k_close": k_close, "k_fb": k_fb, "box_before_close": k_close is not None and k_fb < k_close,
            "k_last": k_last, "k_last_env": k_env, "case_last": case, "last_after_close": k_close is not None and k_last > k_close}


def starts_of(pieces):
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos); pos += len(p)
    return starts


def anchor_positions(pieces, ids):
    """(k_after, k_nl, case) as token indices into the response, from the byte pieces; case nl / eos_first / none."""
    starts = starts_of(pieces)
    b = b"".join(pieces)
    end = first_boxed_end(b)
    if end is None:
        return None
    tok_of = lambda byte: bisect.bisect_right(starts, byte) - 1   # the token holding that byte (never an empty piece)
    k_box = tok_of(end - 1)
    k_after = k_box + 1 if k_box + 1 < len(ids) else None
    nl = b.find(b"\n", end)
    eos = next((k for k in range(k_box + 1, len(ids)) if ids[k] in EOS_IDS), None)
    if nl >= 0:
        k_nl = tok_of(nl)
        if eos is not None and eos < k_nl:
            return k_after, eos, "eos_first"
        return k_after, k_nl, "nl"
    if eos is not None:
        return k_after, eos, "eos_first"
    return k_after, None, "none"


def markers(pieces, k_box_end_byte):
    """(continuation tokens, marker tokens) after the anchor (special tokens excluded)."""
    starts = starts_of(pieces)
    b = b"".join(pieces)
    cont = [k for k, s in enumerate(starts) if s >= k_box_end_byte and len(pieces[k])]
    hit = set()
    for m in re.finditer(MARKERS, b[k_box_end_byte:]):
        lo, hi = k_box_end_byte + m.start(), k_box_end_byte + m.end()
        hit.update(k for k in cont if starts[k] < hi and starts[k] + len(pieces[k]) > lo)
    return len(cont), len(hit)


def token_probs(model, prompt_ids, resp_ids, ks, tid):
    """v2.5.7z: {k: p(tid)} of the distributions that chose resp_ids[k], one forward up to max(ks)."""
    import torch
    ks = [k for k in ks if k is not None]
    if not ks:
        return {}
    ids = torch.tensor([prompt_ids + resp_ids[:max(ks)]], device="cuda")
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[0]
        p = torch.softmax(model.lm_head(h[[len(prompt_ids) + k - 1 for k in ks]]).float(), -1)
    return {k: round(p[i, tid].item(), 6) for i, k in enumerate(ks)}


def stop_probs(model, prompt_ids, resp_ids, ks):
    """{k: (p(<|im_end|>), p(any end token))} of the distributions that chose resp_ids[k], one forward up to max(ks)."""
    import torch
    ks = [k for k in ks if k is not None]
    if not ks:
        return {}
    kmax = max(ks)
    ids = torch.tensor([prompt_ids + resp_ids[:kmax]], device="cuda")
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[0]
        pos = [len(prompt_ids) + k - 1 for k in ks]
        p = torch.softmax(model.lm_head(h[pos]).float(), -1)
    return {k: (round(p[i, EOS_IDS[0]].item(), 6), round(p[i, list(EOS_IDS)].sum().item(), 6)) for i, k in enumerate(ks)}


def describe(vals, by_pid):
    import numpy as np
    v = np.array([x for x in vals if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90])
    out = {"n": int(len(v)), "mean": round(float(v.mean()), 5), "median": round(float(q[2]), 5),
           "p10": round(float(q[0]), 5), "p25": round(float(q[1]), 5), "p75": round(float(q[3]), 5), "p90": round(float(q[4]), 5)}
    pm = np.array([np.mean(x) for x in by_pid.values() if x])
    if len(pm):
        boots = pm[np.random.default_rng(0).integers(0, len(pm), (10000, len(pm)))].mean(1)
        out["mean_ci95_problems"] = [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)]
    return out


def paired_metric(A, B):
    """A, B: {pid: [values]}; difference of per-problem means over common problems, 95% bootstrap CI."""
    import numpy as np
    pids = sorted(p for p in set(A) & set(B) if A[p] and B[p])
    if not pids:
        return {"problems": 0}
    d = np.array([np.mean(A[p]) - np.mean(B[p]) for p in pids])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {"problems": len(pids), "diff": round(float(d.mean()), 5),
            "ci95": [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)]}


def prompts(tok, rows, think, suffix):
    """{pid: prompt token ids} as gen_eval renders them (think: the thinking template)."""
    out = {}
    for r in rows:
        msgs = [{"role": "user", "content": r["problem"] + suffix}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        out[r["pid"]] = tok(s, add_special_tokens=False)["input_ids"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--max_per_model", type=int, default=0)
    ap.add_argument("--no_teacher_nothink", action="store_true")
    ap.add_argument("--no_teacher", action="store_true")   # v2.5.7m (PROBE-TERM3): the teacher's values depend on the text only
    ap.add_argument("--term4", action="store_true")        # v2.5.7z (PROBE-TERM4)
    ap.add_argument("--cache_dir", default=None)   # [v2.5.9z61] per-model results kept for the next run (rerun after a new row)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gen_eval import PROMPT_SUFFIX, load_bench
    rows = load_bench("math500")
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    teacher = None
    if not a.no_teacher:
        ttok = AutoTokenizer.from_pretrained(a.teacher); teacher = load(a.teacher)
        tp_think, tp_plain = prompts(ttok, rows, True, PROMPT_SUFFIX), prompts(ttok, rows, False, PROMPT_SUFFIX)
    report = {"method": __doc__.split("Usage:")[0].strip(), "markers": MARKERS.decode(), "models": {}, "pairs": {}}
    per = {}
    sf = open(a.samples, "w") if a.samples else None
    code = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

    def weights(d):   # [ops-lead's A1 condition] the weights' sizes and mtimes: a result: model fetched again keeps its path
        if not os.path.isdir(d):
            return d
        return sorted((n, os.path.getsize(os.path.join(d, n)), int(os.path.getmtime(os.path.join(d, n)))) for n in os.listdir(d)
                      if n.endswith((".safetensors", ".bin", ".json", ".txt", ".model")) and os.path.isfile(os.path.join(d, n)))
    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)
    for line in open(a.inputs):
        f = line.rstrip("\n").split("\t")
        if len(f) < 5 or line.startswith("#"):
            continue
        tag, mdir, gens, mid, src = f[:5]
        info = {"model": mid, "generations": src}
        if mdir == "-" or not os.path.isfile(gens):
            info["error"] = src if mdir == "-" else "generations missing"; report["models"][tag] = info; continue
        # [v2.5.9z61] (ops-lead 10-22: a rerun for one new row measured every model again, ~0.75 GPU-h) the same tag, model path
        # and weights (names, sizes, mtimes), generations (path, size, mtime), template, prompt suffix, options and this file's
        # sha256: the last run's result, its per-problem values and its samples are used as they are (info["source"] says which)
        st = os.stat(gens)
        key = json.dumps([tag, mdir, weights(mdir), gens, st.st_size, int(st.st_mtime), len(f) > 5 and f[5] == "think", PROMPT_SUFFIX,
                          a.term4, a.no_teacher, a.no_teacher_nothink, a.max_per_model, a.teacher if not a.no_teacher else None, code])
        cf = os.path.join(a.cache_dir, re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".json") if a.cache_dir else None
        if cf and os.path.exists(cf):
            try:
                c = json.load(open(cf))
            except (OSError, ValueError):
                c = {}
            try:   # [ops-lead's A1 condition] the same key but a cache missing a field: measured again, never a crash
                if c.get("key") == key:
                    rinfo = dict(c["info"], reused=True, source=f"reused (measured {c['info']['measured_utc']})")
                    rper = {k: {int(pid): v for pid, v in d.items()} for k, d in c["per"].items()}
                    rsamples = [json.dumps(row) for row in c["samples"]]
                    report["models"][tag] = rinfo; per[tag] = rper
                    for row in rsamples:
                        if sf:
                            sf.write(row + "\n")
                    print(json.dumps({"tag": tag, "reused": True}), flush=True)
                    json.dump(report, open(a.out, "w"), indent=1)
                    continue
            except (KeyError, TypeError, ValueError, AttributeError):
                pass
        rows_out = []
        tok = AutoTokenizer.from_pretrained(mdir); student = load(mdir)
        think = len(f) > 5 and f[5] == "think"   # v2.5.7z: the model's own generation template
        info["template"] = "thinking" if think else "non-thinking"
        sp = prompts(tok, rows, think, PROMPT_SUFFIX)
        recs = [json.loads(x) for x in open(gens)]
        if a.max_per_model:
            recs = recs[:a.max_per_model]
        cnt = defaultdict(int); m = defaultdict(lambda: defaultdict(list)); allv = defaultdict(list); pooled = [0, 0]
        for r in recs:
            ids = r["token_ids"]; cnt["generations"] += 1; cnt["truncated"] += r["finish_reason"] == "length"
            pieces = token_bytes(tok, ids)
            pos = anchor_positions(pieces, ids)
            row = {"tag": tag, "pid": r["pid"], "sample": r["sample"], "correct": r["correct"], "n_tokens": r["n_tokens"],
                   "finish_reason": r["finish_reason"]}
            if pos is None:
                cnt["no_anchor"] += 1
            else:
                k_after, k_nl, case = pos; cnt[case] += 1; row["case"] = case
                end = first_boxed_end(b"".join(pieces))
                ct, mt = markers(pieces, end)
                row.update(cont_tokens=ct, marker_tokens=mt); pooled[0] += mt; pooled[1] += ct
                vals = {"cont_tokens": ct, "marker_frac": (mt / ct) if ct else None}
                k_box, k_env, case_env, ctx = env_positions(pieces, ids); cnt[case_env] += 1   # v2.5.7k
                row.update(case_env=case_env, ctx_env=ctx)
                k16 = list(range(k_box + 1, min(k_box + 17, len(ids))))
                mx = lambda d: max((d[k][0] for k in k16 if k in d), default=None)
                s = stop_probs(student, sp[r["pid"]], ids, [k_after, k_nl, k_env] + k16)
                vals["p_stop_nl"] = s.get(k_nl, (None,))[0]; vals["p_stop_any_nl"] = s.get(k_nl, (None, None))[1]
                vals["p_stop_after_box"] = s.get(k_after, (None,))[0]
                vals["p_stop_env_nl"] = s.get(k_env, (None,))[0]; vals["p_stop_max16"] = mx(s)
                t4 = term4_positions(pieces, ids) if a.term4 else None   # v2.5.7z
                if a.term4 and t4 is None:   # [v2.5.9z53] an anchor but no final box (term4_positions: None): counted, no TERM4 measures
                    cnt["term4_no_box"] += 1   # (10-22 12:41 and 13:46 UTC: PROBE-TERM4 died on t4["k_fb"] of None)
                if t4 is not None:
                    kf16 = list(range(t4["k_fb"] + 1, min(t4["k_fb"] + 17, len(ids))))
                    kl16 = list(range(t4["k_last"] + 1, min(t4["k_last"] + 17, len(ids))))
                    pc = token_probs(student, sp[r["pid"]], ids, kf16, THINK_CLOSE)
                    sl = stop_probs(student, sp[r["pid"]], ids, [t4["k_last_env"]] + kl16)
                    vals["p_close_think_max16"] = max((pc[k] for k in kf16 if k in pc), default=None)
                    vals["p_stop_last_env_nl"] = sl.get(t4["k_last_env"], (None,))[0]
                    vals["p_stop_last_max16"] = max((sl[k][0] for k in kl16 if k in sl), default=None)
                    vals["box_before_close"] = float(t4["box_before_close"]); vals["writes_close"] = float(t4["k_close"] is not None)
                    vals["last_after_close"] = float(t4["last_after_close"])
                    cnt["last_" + t4["case_last"]] += 1
                t = stop_probs(teacher, tp_think[r["pid"]], ids, [k_nl, k_env] + k16) if teacher is not None else {}
                vals["teacher_stop_nl"] = t.get(k_nl, (None,))[0]
                vals["teacher_stop_env_nl"] = t.get(k_env, (None,))[0]; vals["teacher_stop_max16"] = mx(t)
                if teacher is not None and not a.no_teacher_nothink:
                    tn = stop_probs(teacher, tp_plain[r["pid"]], ids, [k_nl, k_env])
                    vals["teacher_nothink_stop_nl"] = tn.get(k_nl, (None,))[0]; vals["teacher_nothink_stop_env_nl"] = tn.get(k_env, (None,))[0]
                for k, v in vals.items():
                    allv[k].append(v)
                    if v is not None:
                        m[k][r["pid"]].append(v)
                row.update(vals)
                for k in ("p_stop_nl", "p_stop_env_nl", "p_stop_max16"):   # (1) split by how the generation ended
                    if vals[k] is not None:
                        allv[k + ("_truncated" if r["finish_reason"] == "length" else "_terminated")].append(vals[k])
            if sf:
                sf.write(json.dumps(row) + "\n")
            rows_out.append(row)
        n = max(cnt["generations"], 1)
        info.update(generations=cnt["generations"], trunc_rate=round(cnt["truncated"] / n, 4), anchor_rate=round(1 - cnt["no_anchor"] / n, 4),
                    cases={c: cnt[c] for c in ("nl", "eos_first", "none", "no_anchor", "env_nl", "env_eos", "env_none", "last_env_nl", "last_env_eos", "last_env_none")},
                    marker_frac_pooled=round(pooled[0] / pooled[1], 5) if pooled[1] else None,
                    measures={k: describe(v, m.get(k, {})) for k, v in allv.items()})
        info.update(measured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source="measured")
        report["models"][tag] = info; per[tag] = m
        if cf:   # [v2.5.9z61]
            json.dump({"key": key, "info": info, "per": {k: {str(pid): v for pid, v in d.items()} for k, d in m.items()}, "samples": rows_out},
                      open(cf + ".tmp", "w")); os.replace(cf + ".tmp", cf)
        print(json.dumps({"tag": tag, "trunc_rate": info["trunc_rate"], **{k: info["measures"][k].get("mean") for k in ("p_stop_nl", "p_stop_env_nl", "p_stop_max16", "marker_frac", "teacher_stop_env_nl") if k in info["measures"]}}), flush=True)
        del student; torch.cuda.empty_cache()
        json.dump(report, open(a.out, "w"), indent=1)   # partial report after every model
    groups = defaultdict(list)
    for tag in per:
        groups[tag.split("_")[0]].append(tag)
    for g, tags in groups.items():
        for i, x in enumerate(tags):
            for y in tags[i + 1:]:
                report["pairs"][f"{x} - {y}"] = {k: paired_metric(per[x].get(k, {}), per[y].get(k, {}))
                                                 for k in ("p_stop_nl", "p_stop_env_nl", "p_stop_max16", "marker_frac", "teacher_stop_nl",
                                                           "teacher_stop_env_nl", "teacher_stop_max16", "cont_tokens",
                                                           "p_close_think_max16", "p_stop_last_env_nl", "p_stop_last_max16")}
    json.dump(report, open(a.out, "w"), indent=1)
    if sf:
        sf.close()


if __name__ == "__main__":
    main()
