#!/usr/bin/env python3
"""[v2.5.9z18] PROBE-WDIFF's math (the reviewer's Q-031: where in the network each method changes the weights, and whether the
changes are low-rank). Delta = M - base for every model of models.tsv, CPU only, nothing downloaded. Tensors are read one
name at a time and in row chunks (safetensors safe_open / get_slice, at most WD_CHUNK elements per model, float64), so the
memory stays near a few GB with 4B models. Per tensor and model: |Delta|, |Delta| / |W|, the share of elements left exactly
unchanged, max |Delta|; per pair of models the cosine of their Deltas; per matrix the spectrum of Delta from its Gram matrix
(Delta^T Delta summed over the row chunks, or Delta Delta^T when it has fewer rows than columns; eigvalsh in float64 gives
the squared singular values, as a full SVD would, without holding Delta): the share of |Delta|^2 in the top 10 singular
values, the top one's share, the stable rank |Delta|^2 / s1^2 and k90 (the fewest singular values holding 90 %). A
training's fp32 save (source local_fp32) is read in bf16, as run_queue_eval.sh reads it for the evaluations and as its bf16
upload holds it, so the unchanged shares compare across models (WD_AS_EVALUATED=0: in fp32).
Checks in the report: torch.linalg.svdvals on layer WD_SVD_CHECK_LAYER's matrices against the Gram route (both timed), and a
null per matrix shape (the top-10 share of an i.i.d. Gaussian matrix; WD_NULL=0 skips it).
Usage: python wdiff.py <models.tsv> <report.json>; models.tsv lines key<TAB>spec<TAB>path<TAB>source, the base first; an
empty path is a model that was not found (its row carries the error). summary.json is written next to report.json."""
import contextlib, json, math, os, re, resource, statistics, sys, time

import torch
from safetensors import safe_open

TOPK = 10
CHUNK = int(os.environ.get("WD_CHUNK", str(1 << 24)))
CHECK_LAYER = int(os.environ.get("WD_SVD_CHECK_LAYER", "0"))
NULL = os.environ.get("WD_NULL", "1") == "1"
AS_EVAL = os.environ.get("WD_AS_EVALUATED", "1") == "1"   # an fp32 local save in bf16, as the evaluations read it (WD_AS_EVALUATED=0: fp32)
torch.set_num_threads(int(os.environ.get("WD_THREADS", "8")))
PROJ = ("q", "k", "v", "o", "gate", "up", "down")
SUB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
       "q_norm": "q_norm", "k_norm": "k_norm", "input_layernorm": "in_norm", "post_attention_layernorm": "post_norm"}
TOP = {"model.embed_tokens.weight": "embed", "lm_head.weight": "lm_head", "model.norm.weight": "final_norm"}
ORDER = {t: i for i, t in enumerate(("embed",) + PROJ + ("q_norm", "k_norm", "in_norm", "post_norm", "final_norm", "lm_head"))}
ARCH = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "tie_word_embeddings", "rope_theta", "max_position_embeddings", "torch_dtype")
PEAK = {"anon_kb": 0}


def kind(name):
    m = re.fullmatch(r"model\.layers\.(\d+)\.(?:self_attn\.|mlp\.)?([a-z_]+)\.weight", name)
    return (int(m.group(1)), SUB.get(m.group(2), m.group(2))) if m else (None, TOP.get(name, name))


def order(name):
    layer, typ = kind(name)
    return (-1 if typ == "embed" else layer if layer is not None else 10 ** 6, ORDER.get(typ, 99), name)


def sym(p):   # the longest of the roots as a variable: no literal node path in the report
    best = None
    for var in ("CKPT_ROOT", "OPD_ROOT", "HF_HOME", "HF_HUB_CACHE", "HOME"):
        v = (os.environ.get(var) or "").rstrip("/")
        if v and (p == v or p.startswith(v + "/")) and (best is None or len(v) > len(best[1])):
            best = (var, v)
    return "$" + best[0] + p[len(best[1]):] if best else p


def mem():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                PEAK["anon_kb"] = max(PEAK["anon_kb"], int(line.split()[1]))
    except OSError:
        pass


def open_model(path, stack):
    """-> (tensor name -> safe_open handle, config); ValueError when shards are missing"""
    idx = os.path.join(path, "model.safetensors.index.json")
    files = (sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx)
             else sorted(f for f in os.listdir(path) if f.endswith(".safetensors")))
    miss = [f for f in files if not os.path.exists(os.path.join(path, f))]
    if miss or not files:
        raise ValueError(f"incomplete: {len(miss)} of {len(files)} shards missing ({', '.join(miss[:3])})" if miss else "no safetensors")
    h = {}
    for f in files:
        fh = stack.enter_context(safe_open(os.path.join(path, f), framework="pt"))
        for k in fh.keys():
            h[k] = fh
    cp = os.path.join(path, "config.json")
    return h, (json.load(open(cp)) if os.path.exists(cp) else {})


def src(h, cfg, name):   # a tied lm_head is the embedding (the 4B configs tie them; a save may still carry lm_head)
    if name in h:
        return name
    if name == "lm_head.weight" and cfg.get("tie_word_embeddings") and "model.embed_tokens.weight" in h:
        return "model.embed_tokens.weight"
    return None


def spectrum(g, tot):
    """eigenvalues of a Gram matrix = squared singular values of Delta -> shares of tot = |Delta|_F^2"""
    t = time.time()
    ev = torch.linalg.eigvalsh(g).flip(0).clamp(min=0)
    dt = time.time() - t
    if tot <= 0:
        return {"top1_share": None, "top10_share": None, "stable_rank": None, "k90": None}, dt
    c = torch.cumsum(ev, 0)
    return {"top1_share": ev[0].item() / tot, "top10_share": ev[:TOPK].sum().item() / tot,
            "stable_rank": tot / ev[0].item() if ev[0].item() > 0 else None,
            "k90": min(int((c < 0.9 * tot).sum().item()) + 1, len(ev))}, dt


def _rows(shape, bsl):
    rows = shape[0]; cols = math.prod(shape[1:]) if len(shape) > 1 else 1
    wide = len(shape) == 2 and rows < cols   # fewer rows than columns: one chunk, Gram Delta Delta^T
    step = rows if wide else max(1, CHUNK // max(cols, 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        w = bsl[r0:r1].double()
        yield r0, r1, wide, (w.reshape(r1 - r0, cols) if len(shape) == 2 else w.reshape(-1))


def tensor_stats(bsl, shape, sl, tm, cast=()):
    keys = list(sl); mat = len(shape) == 2
    w2 = 0.0; d2 = dict.fromkeys(keys, 0.0); zero = dict.fromkeys(keys, 0); mx = dict.fromkeys(keys, 0.0); gram = {}
    dots = {(a, b): 0.0 for i, a in enumerate(keys) for b in keys[i + 1:]}
    numel = math.prod(shape)
    for r0, r1, wide, w in _rows(shape, bsl):
        t = time.time()
        ds = {k: ((sl[k][r0:r1].to(torch.bfloat16) if k in cast else sl[k][r0:r1]).double().reshape(w.shape) - w) for k in keys}
        tm["read_s"] += time.time() - t
        wf = w.reshape(-1); w2 += torch.dot(wf, wf).item()
        for k, d in ds.items():
            f = d.reshape(-1)
            d2[k] += torch.dot(f, f).item(); zero[k] += int((f == 0).sum().item()); mx[k] = max(mx[k], f.abs().max().item())
        for a, b in dots:
            dots[(a, b)] += torch.dot(ds[a].reshape(-1), ds[b].reshape(-1)).item()
        if mat:
            t = time.time()
            for k, d in ds.items():
                g = d @ d.T if wide else d.T @ d
                if k in gram:
                    gram[k] += g
                else:
                    gram[k] = g
            tm["gram_s"] += time.time() - t
        mem(); del ds   # RssAnon while the chunk's Deltas are alive (maxrss also counts the mapped shards)
    out = {"numel": numel, "W_norm": math.sqrt(w2), "delta": {}, "cos": {}}
    for k in keys:
        e = {"norm": math.sqrt(d2[k]), "rel": math.sqrt(d2[k] / w2) if w2 > 0 else None, "frac_unchanged": zero[k] / numel, "max_abs": mx[k]}
        if mat:
            s, dt = spectrum(gram.pop(k), d2[k]); tm["eig_s"] += dt; e.update(s)
        out["delta"][k] = e
    for (a, b), v in dots.items():
        out["cos"][f"{a}|{b}"] = v / math.sqrt(d2[a] * d2[b]) if d2[a] > 0 and d2[b] > 0 else None
    return out


def null_share(shape, tm):
    """top-10 share of an i.i.d. Gaussian matrix of this shape (seed 0), by the same Gram route, generated in row chunks"""
    t = time.time(); gen = torch.Generator().manual_seed(0)
    rows, cols = shape; wide = rows < cols; step = rows if wide else max(1, CHUNK // cols); g = None; tot = 0.0
    for r0 in range(0, rows, step):
        x = torch.randn(min(rows, r0 + step) - r0, cols, generator=gen, dtype=torch.float64)
        tot += torch.dot(x.reshape(-1), x.reshape(-1)).item()
        y = x @ x.T if wide else x.T @ x
        g = y if g is None else g + y
    s, _ = spectrum(g, tot); tm["null_s"] += time.time() - t
    return s["top10_share"]


def thirds(xs):
    n = len(xs); sizes = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]; out = {}; i = 0
    for name, s in zip(("early", "middle", "late"), sizes):
        part = [x for x in xs[i:i + s] if x is not None]; out[name] = statistics.fmean(part) if part else None; i += s
    return out


def summarize(rows, models, null):
    summ, line = {}, []
    for m in models:
        k = m["key"]
        if m["error"]:
            summ[k] = {"error": m["error"]}; line.append(f"{k} missing"); continue
        R = [r for r in rows if k in (r.get("delta") or {})]
        if not R or not sum(r["W_norm"] ** 2 for r in R):
            summ[k] = {"error": "no tensor in common with the base"}; line.append(f"{k} no common tensor"); continue
        P = [r for r in R if r["type"] in PROJ]
        d2 = sum(r["delta"][k]["norm"] ** 2 for r in R); w2 = sum(r["W_norm"] ** 2 for r in R)
        layers = sorted({r["layer"] for r in P})
        by_layer = [statistics.fmean([r["delta"][k]["rel"] for r in P if r["layer"] == l and r["delta"][k]["rel"] is not None] or [0.0]) for l in layers]
        by_type = {}
        for t in sorted({r["type"] for r in R}, key=lambda t: ORDER.get(t, 99)):
            T = [r for r in R if r["type"] == t]
            sh = [r["delta"][k]["top10_share"] for r in T if r["delta"][k].get("top10_share") is not None]
            nl = [null.get("x".join(map(str, r["shape"]))) for r in T if r["delta"][k].get("top10_share") is not None]
            by_type[t] = {"n": len(T), "rel_mean": statistics.fmean([r["delta"][k]["rel"] or 0.0 for r in T]),
                          "rel_agg": math.sqrt(sum(r["delta"][k]["norm"] ** 2 for r in T) / max(sum(r["W_norm"] ** 2 for r in T), 1e-300)),
                          "frac_unchanged": sum(r["delta"][k]["frac_unchanged"] * r["numel"] for r in T) / sum(r["numel"] for r in T),
                          "top10_share_mean": statistics.fmean(sh) if sh else None,
                          "null_top10_share_mean": statistics.fmean(nl) if nl and None not in nl else None}
        sh = [r["delta"][k]["top10_share"] for r in P if r["delta"][k].get("top10_share") is not None]
        nl = [null.get("x".join(map(str, r["shape"]))) for r in P if r["delta"][k].get("top10_share") is not None]
        k90 = [r["delta"][k]["k90"] for r in P if r["delta"][k].get("k90") is not None]
        sr = [r["delta"][k]["stable_rank"] for r in P if r["delta"][k].get("stable_rank") is not None]
        ranked = sorted(by_type, key=lambda t: -by_type[t]["rel_mean"])
        s = {"rel_overall": math.sqrt(d2 / w2) if w2 else None, "rel_by_layer": by_layer, "rel_by_third": thirds(by_layer),
             "peak_layer": layers[max(range(len(by_layer)), key=lambda i: by_layer[i])] if by_layer else None,
             "types_by_rel": [[t, by_type[t]["rel_mean"]] for t in ranked],
             "proj_types_by_rel": [[t, by_type[t]["rel_mean"]] for t in ranked if t in PROJ],
             "frac_unchanged": sum(r["delta"][k]["frac_unchanged"] * r["numel"] for r in R) / max(sum(r["numel"] for r in R), 1),
             "top10_share_mean": statistics.fmean(sh) if sh else None,
             "null_top10_share_mean": statistics.fmean(nl) if nl and None not in nl else None,
             "k90_median": statistics.median(k90) if k90 else None, "stable_rank_median": statistics.median(sr) if sr else None,
             "by_type": by_type}
        s["lowrank_x_null"] = s["top10_share_mean"] / s["null_top10_share_mean"] if s["top10_share_mean"] is not None and s["null_top10_share_mean"] else None
        summ[k] = s
        top = s["proj_types_by_rel"][0][0] if s["proj_types_by_rel"] else None
        line.append(f"{k} rel {s['rel_overall']:.2g}, most {top}, top10 {s['top10_share_mean'] or 0:.2f}"
                    + (f" ({s['lowrank_x_null']:.1f}x null)" if s["lowrank_x_null"] else ""))
    return summ, "; ".join(line)


def cos_agg(rows, pairs, pick):
    out = {}
    for p in pairs:
        a, b = p.split("|"); num = da = db = 0.0
        for r in rows:   # the cosine of the Deltas concatenated over the rows both have (a zero Delta adds 0 to the dot)
            if not pick(r) or a not in (r.get("delta") or {}) or b not in r["delta"]:
                continue
            na, nb = r["delta"][a]["norm"], r["delta"][b]["norm"]; num += (r["cos"].get(p) or 0.0) * na * nb; da += na * na; db += nb * nb
        out[p] = num / math.sqrt(da * db) if da > 0 and db > 0 else None
    return out


def main(tsv, out):
    t0 = time.time(); tm = {"read_s": 0.0, "gram_s": 0.0, "eig_s": 0.0, "null_s": 0.0, "svd_check_s": 0.0}
    lines = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
    (bkey, bspec, bpath, bsrc), rest = lines[0][:4], [l[:4] for l in lines[1:]]
    rep = {"task": "PROBE-WDIFF", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "chunk_elems": CHUNK,
           "method": "Delta = M - base in float64 by row chunks; spectrum = eigvalsh of the float64 Gram matrix (exact squared singular values)",
           "base": {"key": bkey, "spec": bspec, "path": sym(bpath) if bpath else None, "source": bsrc or None, "error": None},
           "models": [], "rows": [], "error": None}
    with contextlib.ExitStack() as stack:
        try:
            if not bpath:
                raise ValueError(f"not found: {bspec} has no local copy (PROBE-WDIFF never downloads)")
            bh, bcfg = open_model(bpath, stack)
        except Exception as e:
            rep["base"]["error"] = rep["error"] = f"base: {str(e)[:300]}"
            json.dump(rep, open(out, "w"), indent=1); return
        opened, cast = [], set()
        for key, spec, path, source in rest:
            m = {"key": key, "spec": spec, "path": sym(path) if path else None, "source": source or None, "error": None,
                 "dtype": None, "cast": None, "config_diff": {}}
            try:
                if not path:
                    raise ValueError(f"not found: {spec} has no local copy (PROBE-WDIFF never downloads)")
                h, cfg = open_model(path, stack)
                m["dtype"] = sorted({h[n].get_slice(n).get_dtype() for n in list(h)[:50]})
                m["config_diff"] = {a: [bcfg.get(a), cfg.get(a)] for a in ARCH if bcfg.get(a) != cfg.get(a)}
                if source == "local_fp32" and AS_EVAL:
                    m["cast"] = "bfloat16 (as evaluated)"; cast.add(key)
                opened.append((key, h, cfg))
            except Exception as e:
                m["error"] = str(e)[:300]
            rep["models"].append(m)
        rep["base"]["dtype"] = sorted({bh[n].get_slice(n).get_dtype() for n in list(bh)[:50]})
        names = set(bh)
        if any("lm_head.weight" in h for _, h, _ in opened):
            names.add("lm_head.weight")
        rep["only_in_models"] = {k: sorted(set(h) - names)[:50] for k, h, _ in opened if set(h) - names}
        checks, null = [], {}
        for name in sorted(names, key=order):
            layer, typ = kind(name); row = {"name": name, "layer": layer, "type": typ}
            bn = src(bh, bcfg, name)
            if bn is None:
                row["error"] = "not in the base"; rep["rows"].append(row); continue
            bsl = bh[bn].get_slice(bn); shape = list(bsl.get_shape()); row["shape"] = shape
            sl, errs = {}, {}
            for key, h, cfg in opened:
                mn = src(h, cfg, name)
                if mn is None:
                    errs[key] = "missing in the model"; continue
                s = h[mn].get_slice(mn)
                if list(s.get_shape()) != shape:
                    errs[key] = f"shape {list(s.get_shape())} vs the base's {shape}"; continue
                sl[key] = s
            if not shape:
                row["error"] = "a scalar, skipped"; rep["rows"].append(row); continue
            try:
                row.update(tensor_stats(bsl, shape, sl, tm, cast)); row["errors"] = errs or None
            except Exception as e:   # a read error: this row's, the run goes on
                row["error"] = f"{type(e).__name__}: {str(e)[:300]}"; rep["rows"].append(row); continue
            if len(shape) == 2 and (typ in PROJ or typ in ("embed", "lm_head")):
                if NULL and "x".join(map(str, shape)) not in null:
                    null["x".join(map(str, shape))] = null_share(shape, tm)
                if layer == CHECK_LAYER and typ in PROJ and sl:   # the full SVD on one Delta of this layer's matrices, timed
                    k = next(iter(sl)); t = time.time()
                    x = sl[k][:]; sv = torch.linalg.svdvals((x.to(torch.bfloat16) if k in cast else x).double() - bsl[:].double())
                    dt = time.time() - t; tm["svd_check_s"] += dt; e = sv ** 2
                    ref = (e[:TOPK].sum() / e.sum()).item() if e.sum() > 0 else None; got = row["delta"][k]["top10_share"]
                    checks.append({"name": name, "key": k, "gram_top10": got, "svd_top10": ref, "svd_s": round(dt, 3),
                                   "abs_diff": abs(got - ref) if got is not None and ref is not None else None})
            rep["rows"].append(row); mem()
        tm["total_s"] = time.time() - t0
    pairs = [f"{a}|{b}" for i, (a, _, _) in enumerate(opened) for b, _, _ in opened[i + 1:]]
    rep["null_top10_share"] = null; rep["svd_check"] = checks
    rep["svd_check_max_abs_diff"] = max((c["abs_diff"] for c in checks if c["abs_diff"] is not None), default=None)
    rep["cos_overall"] = cos_agg(rep["rows"], pairs, lambda r: True)
    rep["cos_by_type"] = {t: cos_agg(rep["rows"], pairs, lambda r, t=t: r["type"] == t) for t in sorted({r["type"] for r in rep["rows"]}, key=lambda t: ORDER.get(t, 99))}
    nl = sorted({r["layer"] for r in rep["rows"] if r.get("layer") is not None})
    rep["cos_by_layer"] = {p: [cos_agg(rep["rows"], [p], lambda r, l=l: r.get("layer") == l and r["type"] in PROJ)[p] for l in nl] for p in pairs}
    rep["summary"], rep["summary_line"] = summarize(rep["rows"], rep["models"], null)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rep["timing"] = {k: round(v, 2) for k, v in tm.items()}
    rep["peak"] = {"maxrss_gb": round(ru / (1e9 if sys.platform == "darwin" else 1e6), 2),
                   "anon_gb": round(PEAK["anon_kb"] / 1e6, 2) if PEAK["anon_kb"] else None}
    json.dump(rep, open(out, "w"), indent=1)
    keep = ("utc", "base", "models", "summary_line", "summary", "cos_overall", "cos_by_type", "cos_by_layer", "null_top10_share",
            "svd_check_max_abs_diff", "timing", "peak", "method")
    json.dump({k: rep[k] for k in keep}, open(os.path.join(os.path.dirname(out) or ".", "summary.json"), "w"), indent=1)
    print(rep["summary_line"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
