#!/usr/bin/env bash
# [v2.5.9z18] Stub tests of probe_wdiff.sh + wdiff.py: tiny random Qwen3-shaped safetensors models (2 layers, hidden 32) in each
# place the probe looks (a local bf16 save next to a decoy fp32 one, a local fp32 save under $CKPT_ROOT, the download cache,
# a Hugging Face cache snapshot, $OPD_ROOT/models), every number of the report against numpy (float64, np.linalg.svd), the
# missing / incomplete / no-base rows. Needs a python with torch, safetensors and numpy (the node's venv; on the Mac set
# WD_TEST_PY). Usage: [WD_TEST_PY=<python>] bash scripts/test_probe_wdiff.sh -> PROBE_WDIFF_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0; PY=${WD_TEST_PY:-python3}
$PY -c "import torch, safetensors, numpy" 2> /dev/null || { echo "PROBE_WDIFF_TEST SKIP (no torch / safetensors / numpy in $PY: set WD_TEST_PY)"; exit 3; }
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/opd $T/ckpt $T/hf $T/bin; cd $T
  cp $SRC/scripts/probe_wdiff.sh $SRC/scripts/wdiff.py $T/rb/scripts/
  { echo "export OPD_ROOT=$T/opd RUNBOOK=$T/rb CKPT_ROOT=$T/ckpt HF_HOME=$T/hf STUDENT_4B=$T/opd/models/Qwen3-4B"; grep '^has_weights()' $SRC/scripts/env.sh; } > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  command -v timeout > /dev/null || { printf '#!/usr/bin/env bash\n[ "$1" = -s ] && shift 2; shift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout; }
  export CALLS=$T/calls PATH=$T/bin:$PATH WD_PY=$PY WD_NICE=0; : > $CALLS
}
teardown() { cd $SRC; rm -rf $T; }
# the models: P5 a rank-2 Delta per matrix (sharded, with an index), P5self dense on the matrices and norms left exactly as
# they are (fp32), RL sparse (70 % unchanged) and strongest on down_proj, saved with an explicit lm_head; T2507 dense, its
# own rope_theta and a k_norm of another shape in layer 1; I2507 absent (or, with "incomplete", one shard of two missing)
gen() {
  $PY - "$T" "$@" <<'P'
import json, os, sys
import torch
from safetensors.torch import save_file
T, what = sys.argv[1], sys.argv[2:]
g = torch.Generator().manual_seed(7)
L, H, NH, KV, HD, I, V = 2, 32, 4, 2, 8, 48, 64
cfg = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L, "num_attention_heads": NH,
       "num_key_value_heads": KV, "head_dim": HD, "vocab_size": V, "tie_word_embeddings": True, "rope_theta": 1000000, "torch_dtype": "bfloat16"}
shapes = {"model.embed_tokens.weight": [V, H], "model.norm.weight": [H]}
for l in range(L):
    p = f"model.layers.{l}."
    shapes.update({p + "self_attn.q_proj.weight": [NH * HD, H], p + "self_attn.k_proj.weight": [KV * HD, H], p + "self_attn.v_proj.weight": [KV * HD, H],
                   p + "self_attn.o_proj.weight": [H, NH * HD], p + "self_attn.q_norm.weight": [HD], p + "self_attn.k_norm.weight": [HD],
                   p + "mlp.gate_proj.weight": [I, H], p + "mlp.up_proj.weight": [I, H], p + "mlp.down_proj.weight": [H, I],
                   p + "input_layernorm.weight": [H], p + "post_attention_layernorm.weight": [H]})
rn = lambda *s: torch.randn(*s, generator=g)
base = {k: (1 + 0.1 * rn(*s) if len(s) == 1 else 0.05 * rn(*s)).to(torch.bfloat16) for k, s in shapes.items()}
def save(d, sd, c=cfg, shards=1):
    os.makedirs(d, exist_ok=True); json.dump(c, open(os.path.join(d, "config.json"), "w"))
    if shards == 1:
        save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(d, "model.safetensors")); return
    ks = sorted(sd); parts = [ks[i::shards] for i in range(shards)]; wm = {}
    for i, part in enumerate(parts):
        f = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"; save_file({k: sd[k].contiguous() for k in part}, os.path.join(d, f)); wm.update({k: f for k in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
if "nobase" not in what:
    save(f"{T}/opd/models/Qwen3-4B", base)
if "main" in what:
    W = {k: v.double() for k, v in base.items()}
    p5 = {k: (W[k] + (0.5 * W[k].std() * (rn(s[0], 2) @ rn(2, s[1])).double() / 2 if len(s) == 2 else 0.01 * rn(*s).double())).to(torch.bfloat16) for k, s in shapes.items()}
    save(f"{T}/opd/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16", p5, shards=2)
    save(f"{T}/opd/runs/E10d-P5/ckpt/global_step_150/actor/huggingface", {k: v.float() for k, v in base.items()})   # decoy: the bf16 copy wins
    ps = {k: (W[k] + 0.2 * W[k].std() * rn(*s).double()).float() if len(s) == 2 else base[k].float() for k, s in shapes.items()}
    save(f"{T}/ckpt/runs/E10d-P5self/ckpt/global_step_50/actor/huggingface", ps)
    rl = {}
    for k, s in shapes.items():
        m = (torch.rand(*s, generator=g) < 0.3).double(); amp = 2.0 if "down_proj" in k else 0.3
        rl[k] = (W[k] + amp * W[k].abs().mean() * m * rn(*s).double()).to(torch.bfloat16)
    rl["lm_head.weight"] = rl["model.embed_tokens.weight"].clone()
    save(f"{T}/ckpt/adhoc_models/result/runs/E12d-P0/global_step_100/actor/huggingface", rl)
    t5 = {k: (W[k] + 0.3 * W[k].std() * rn(*s).double()).to(torch.bfloat16) for k, s in shapes.items()}
    t5["model.layers.1.self_attn.k_norm.weight"] = torch.ones(HD + 1, dtype=torch.bfloat16)
    save(f"{T}/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc123", t5, dict(cfg, rope_theta=5000000))
if "incomplete" in what:
    save(f"{T}/opd/models/Qwen3-4B-Instruct-2507", base, shards=2); os.remove(f"{T}/opd/models/Qwen3-4B-Instruct-2507/model-00002-of-00002.safetensors")
P
}
# numpy reference: every row, model and number of report.json (float64; singular values from np.linalg.svd), P5self's fp32
# save in bf16 as the probe reads it by default ("fp32": as stored); prints "<compared> <mismatches>" and the first mismatches
check() {
  $PY - "$T" "$1" "${2:-bf16}" <<'P'
import json, sys
import numpy as np, torch
from safetensors import safe_open
T, rp, mode = sys.argv[1], sys.argv[2], sys.argv[3]
paths = {"base": f"{T}/opd/models/Qwen3-4B", "P5": f"{T}/opd/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16",
         "P5self": f"{T}/ckpt/runs/E10d-P5self/ckpt/global_step_50/actor/huggingface", "RL": f"{T}/ckpt/adhoc_models/result/runs/E12d-P0/global_step_100/actor/huggingface",
         "T2507": f"{T}/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc123"}
import glob
def load(d, bf16=False):
    out = {}
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, framework="pt") as h:
            out.update({k: (h.get_tensor(k).to(torch.bfloat16) if bf16 else h.get_tensor(k)).double().numpy() for k in h.keys()})
    return out
M = {k: load(p, k == "P5self" and mode == "bf16") for k, p in paths.items()}
get = lambda sd, name: sd.get(name, sd.get("model.embed_tokens.weight") if name == "lm_head.weight" else None)
rep = json.load(open(rp)); n = bad = 0; msgs = []
def cmp(what, got, want, tol=1e-9):
    global n, bad
    n += 1
    if (got is None) != (want is None) or (want is not None and abs(got - want) > tol * max(1.0, abs(want))):
        bad += 1; msgs.append(f"{what}: got {got} want {want}")
for r in rep["rows"]:
    W = get(M["base"], r["name"]); D = {}
    for k in ("P5", "P5self", "RL", "T2507"):
        x = get(M[k], r["name"])
        if x is not None and x.shape == W.shape:
            D[k] = x - W
    cmp(r["name"] + " models", len(r.get("delta", {})), len(D), 0)
    cmp(r["name"] + " W_norm", r["W_norm"], float(np.linalg.norm(W)))
    for k, d in D.items():
        e = r["delta"][k]; nd = float(np.linalg.norm(d))
        cmp(f"{r['name']} {k} norm", e["norm"], nd); cmp(f"{r['name']} {k} rel", e["rel"], nd / float(np.linalg.norm(W)))
        cmp(f"{r['name']} {k} frac_unchanged", e["frac_unchanged"], float((d == 0).mean())); cmp(f"{r['name']} {k} max_abs", e["max_abs"], float(np.abs(d).max()))
        if d.ndim == 2:
            s2 = np.linalg.svd(d, compute_uv=False) ** 2; tot = s2.sum()
            cmp(f"{r['name']} {k} top10", e["top10_share"], float(s2[:10].sum() / tot) if tot else None)
            cmp(f"{r['name']} {k} top1", e["top1_share"], float(s2[0] / tot) if tot else None)
            cmp(f"{r['name']} {k} stable_rank", e["stable_rank"], float(tot / s2[0]) if tot else None, 1e-7)
            cmp(f"{r['name']} {k} k90", e["k90"], int(np.searchsorted(np.cumsum(s2), 0.9 * tot) + 1) if tot else None, 0)
    ks = list(D)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            na, nb = np.linalg.norm(D[a]), np.linalg.norm(D[b])
            cmp(f"{r['name']} cos {a}|{b}", r["cos"].get(f"{a}|{b}"), float((D[a] * D[b]).sum() / (na * nb)) if na and nb else None)
# the aggregate cosine over every tensor, and P5's relative change by layer (mean over the layer's seven matrices)
for p in rep["cos_overall"]:
    a, b = p.split("|"); num = da = db = 0.0
    for r in rep["rows"]:
        W = get(M["base"], r["name"]); xa, xb = get(M[a], r["name"]), get(M[b], r["name"])
        if xa is None or xb is None or xa.shape != W.shape or xb.shape != W.shape:
            continue
        num += ((xa - W) * (xb - W)).sum(); da += ((xa - W) ** 2).sum(); db += ((xb - W) ** 2).sum()
    cmp("cos_overall " + p, rep["cos_overall"][p], float(num / np.sqrt(da * db)))
for l in (0, 1):
    rels = [np.linalg.norm(get(M["P5"], r["name"]) - get(M["base"], r["name"])) / np.linalg.norm(get(M["base"], r["name"]))
            for r in rep["rows"] if r.get("layer") == l and r["type"] in ("q", "k", "v", "o", "gate", "up", "down")]
    cmp(f"P5 rel_by_layer {l}", rep["summary"]["P5"]["rel_by_layer"][l], float(np.mean(rels)))
print(n, bad); print("\n".join(msgs[:5]))
P
}
r() { $PY -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/opd/runs/PROBE-WDIFF/report.json "$1"; }
# 1. five models, one absent: every number against numpy, with chunks of 100 elements (the Gram and the sums over chunks)
setup; gen main
WD_CHUNK=100 bash $T/rb/scripts/probe_wdiff.sh cpu > /dev/null 2>&1
ok "a report, summary.json; .done" '[ -s $T/opd/runs/PROBE-WDIFF/report.json ] && [ -s $T/opd/runs/PROBE-WDIFF/summary.json ] && [ -f $T/opd/runs/PROBE-WDIFF/.done ]'
ok "P5self's fp32 save read in bf16 (as evaluated); the bf16 saves as they are" '[ "$(r "[m[\"cast\"] for m in v[\"models\"]]")" = "[None, '"'"'bfloat16 (as evaluated)'"'"', None, None, None]" ]'
ok "each model from its place: base models, P5 local bf16 (not the fp32 decoy), P5self local fp32 under CKPT_ROOT, RL download cache, T2507 HF cache" '[ "$(r "\" \".join([v[\"base\"][\"source\"]] + [m[\"source\"] or \"-\" for m in v[\"models\"]])")" = "models local_bf16 local_fp32 download_cache hf_cache -" ]'
ok "paths symbolic, no literal test path in report.json / summary.json" '[ "$(r "v[\"models\"][0][\"path\"]")" = "\$OPD_ROOT/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16" ] && [ "$(r "v[\"models\"][3][\"path\"]")" = "\$HF_HOME/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc123" ] && ! grep -q "$T" $T/opd/runs/PROBE-WDIFF/report.json $T/opd/runs/PROBE-WDIFF/summary.json'
ok "I2507 absent: a model row with its error, a summary row, no Delta anywhere" '[[ "$(r "v[\"models\"][4][\"error\"]")" == "not found: Qwen/Qwen3-4B-Instruct-2507 has no local copy"* ]] && [ "$(r "v[\"summary\"][\"I2507\"][\"error\"][:9]")" = "not found" ] && [ "$(r "sum(\"I2507\" in (x.get(\"delta\") or {}) for x in v[\"rows\"])")" = 0 ] && grep -q "I2507: Qwen/Qwen3-4B-Instruct-2507 not found (never downloaded)" $T/opd/runs/PROBE-WDIFF/probe.log'
got=$(check $T/opd/runs/PROBE-WDIFF/report.json)
ok "every number against numpy: norms, relative change, unchanged share, max, top-10 / top-1 share, stable rank, k90, pairwise cosines ($(echo $got | head -c 300))" '[ "$(echo "$got" | head -1 | cut -d" " -f2)" = 0 ] && [ "$(echo "$got" | head -1 | cut -d" " -f1)" -gt 600 ]'
ok "rows: 2 layers x 11 tensors + embed + final norm + lm_head; the tied lm_head row (RL saved its own) for all four" '[ "$(r "len(v[\"rows\"])")" = 25 ] && [ "$(r "\" \".join([x[\"type\"] for x in v[\"rows\"]][:3] + [v[\"rows\"][-1][\"name\"], str(len(v[\"rows\"][-1][\"delta\"]))])")" = "embed q k lm_head.weight 4" ]'
ok "T2507's other k_norm shape: an error in that row only, the other three models computed; its rope_theta in config_diff" '[[ "$(r "[x for x in v[\"rows\"] if x[\"name\"] == \"model.layers.1.self_attn.k_norm.weight\"][0][\"errors\"][\"T2507\"]")" == "shape [9] vs the base"* ]] && [ "$(r "len([x for x in v[\"rows\"] if x[\"name\"] == \"model.layers.1.self_attn.k_norm.weight\"][0][\"delta\"])")" = 3 ] && [ "$(r "v[\"models\"][3][\"config_diff\"][\"rope_theta\"]")" = "[1000000, 5000000]" ]'
ok "P5self's norms exactly unchanged: share 1.0, rel 0, cosines None" '[ "$(r "sorted(set([(x[\"delta\"][\"P5self\"][\"frac_unchanged\"], x[\"delta\"][\"P5self\"][\"rel\"], x[\"cos\"][\"P5|P5self\"]) for x in v[\"rows\"] if x[\"type\"].endswith(\"norm\")]))")" = "[(1.0, 0.0, None)]" ]'
ok "summary: RL moved down_proj most and left ~70 % unchanged; P5 rank-2 (top-10 > 0.99, k90 <= 2) more low-rank than T2507" '[ "$(r "v[\"summary\"][\"RL\"][\"proj_types_by_rel\"][0][0]")" = down ] && [ "$(r "0.6 < v[\"summary\"][\"RL\"][\"frac_unchanged\"] < 0.8")" = True ] && [ "$(r "v[\"summary\"][\"P5\"][\"top10_share_mean\"] > 0.99 and v[\"summary\"][\"P5\"][\"k90_median\"] <= 2 and v[\"summary\"][\"P5\"][\"lowrank_x_null\"] > v[\"summary\"][\"T2507\"][\"lowrank_x_null\"]")" = True ]'
ok "summary shape: by layer (2), by third, types ranked by relative change, nulls per shape in (0, 1)" '[ "$(r "\" \".join(str(x) for x in (len(v[\"summary\"][\"P5\"][\"rel_by_layer\"]), \",\".join(sorted(v[\"summary\"][\"P5\"][\"rel_by_third\"])), [t[1] for t in v[\"summary\"][\"RL\"][\"types_by_rel\"]] == sorted([t[1] for t in v[\"summary\"][\"RL\"][\"types_by_rel\"]], reverse=True), \",\".join(sorted(v[\"null_top10_share\"])), all(0 < x < 1 for x in v[\"null_top10_share\"].values())))")" = "2 early,late,middle True 16x32,32x32,32x48,48x32,64x32 True" ]'
ok "the full-SVD check on layer 0 (7 matrices of P5) agrees with the Gram route; timings and peak memory reported" '[ "$(r "\" \".join(str(x) for x in (len(v[\"svd_check\"]), \",\".join(sorted({c[\"key\"] for c in v[\"svd_check\"]})), v[\"svd_check_max_abs_diff\"] < 1e-10, \",\".join(sorted(v[\"timing\"])), v[\"peak\"][\"maxrss_gb\"] > 0))")" = "7 P5 True eig_s,gram_s,null_s,read_s,svd_check_s,total_s True" ]'
ok "private upload of report.json, summary.json, probe.log (not models.tsv); the done event carries the summary line" 'grep -q "^remote put $T/opd/runs/PROBE-WDIFF/up logs/abcd1234/probe_wdiff/PROBE-WDIFF" $CALLS && [ "$(ls $T/opd/runs/PROBE-WDIFF/up | tr "\n" " ")" = "probe.log report.json summary.json " ] && grep -q "^control event PROBE-WDIFF probe_wdiff_done P5 rel .*, most .*, top10 1.00 (.*x null); .*; I2507 missing$" $CALLS'
cp $T/opd/runs/PROBE-WDIFF/report.json $T/first.json; : > $CALLS
bash $T/rb/scripts/probe_wdiff.sh cpu > $T/out2 2>&1
ok ".done: TASK_ALREADY_DONE, nothing run" 'grep -q "TASK_ALREADY_DONE PROBE-WDIFF" $T/out2 && [ ! -s $CALLS ]'
# 2. the rerun (rerun renames .done and report.json) with the default chunk size: the same numbers as with 100-element chunks
mv $T/opd/runs/PROBE-WDIFF/.done $T/opd/runs/PROBE-WDIFF/.done.prev_1; mv $T/opd/runs/PROBE-WDIFF/report.json $T/opd/runs/PROBE-WDIFF/report.json.prev_1
bash $T/rb/scripts/probe_wdiff.sh cpu > /dev/null 2>&1
got=$($PY - $T/first.json $T/opd/runs/PROBE-WDIFF/report.json <<'P'
import json, sys
a, b = (json.load(open(f))["rows"] for f in sys.argv[1:3]); worst = [0.0]
def walk(x, y):
    if isinstance(x, dict):
        assert x.keys() == y.keys(); [walk(x[k], y[k]) for k in x]
    elif isinstance(x, list):
        assert len(x) == len(y); [walk(p, q) for p, q in zip(x, y)]
    elif isinstance(x, float):
        worst[0] = max(worst[0], abs(x - y) / max(1.0, abs(x)))
    else:
        assert x == y, (x, y)
walk(a, b); print(worst[0] < 1e-12)
P
)
ok "rerun with the default chunk: the same rows as with 100-element chunks (relative difference < 1e-12)" '[ "$got" = True ] && [ -f $T/opd/runs/PROBE-WDIFF/.done ]'
mv $T/opd/runs/PROBE-WDIFF/.done $T/opd/runs/PROBE-WDIFF/.done.prev_2; mv $T/opd/runs/PROBE-WDIFF/report.json $T/opd/runs/PROBE-WDIFF/report.json.prev_2
WD_AS_EVALUATED=0 bash $T/rb/scripts/probe_wdiff.sh cpu > /dev/null 2>&1
got=$(check $T/opd/runs/PROBE-WDIFF/report.json fp32)
ok "WD_AS_EVALUATED=0: P5self in fp32, every number against the fp32 reference ($(echo $got | head -c 200))" '[ "$(echo "$got" | head -1 | cut -d" " -f2)" = 0 ] && [ "$(r "v[\"models\"][1][\"cast\"]")" = None ]'
teardown
# 3. an incomplete model (one shard of two) and a spec outside the patterns: rows with their errors, the run still done
setup; gen incomplete
WD_MODELS="I2507=Qwen/Qwen3-4B-Instruct-2507 BAD=result:runs/../x" bash $T/rb/scripts/probe_wdiff.sh cpu > /dev/null 2>&1
ok "an incomplete model and a refused spec are rows with errors; done with nothing to compare" '[[ "$(r "v[\"models\"][0][\"error\"]")" == "incomplete: 1 of 2 shards missing (model-00002-of-00002.safetensors)" ]] && [[ "$(r "v[\"models\"][1][\"error\"]")" == "not found: result:runs/../x"* ]] && [ "$(r "v[\"summary_line\"]")" = "I2507 missing; BAD missing" ] && [ -f $T/opd/runs/PROBE-WDIFF/.done ]'
teardown
# 4. no base: the report carries the error, a failed event, no .done (run_tasks retries; rerun once the base is there)
setup; gen nobase
bash $T/rb/scripts/probe_wdiff.sh cpu > /dev/null 2>&1; rc=$?
ok "no base: exit 1, report error, probe_wdiff_failed, report still uploaded, no .done" '[ $rc = 1 ] && [[ "$(r "v[\"error\"]")" == "base: not found: Qwen/Qwen3-4B has no local copy"* ]] && grep -q "^control event PROBE-WDIFF probe_wdiff_failed base: not found" $CALLS && grep -q "^remote put .*probe_wdiff/PROBE-WDIFF" $CALLS && [ ! -f $T/opd/runs/PROBE-WDIFF/.done ]'
teardown
# 5. registration: task.sh dispatches it, run_tasks.sh runs it in the CPU lane, the heartbeat calls it a probe
got=$( eval "$(sed -n '/^cpu_task() {/p' $SRC/scripts/run_tasks.sh)"; cpu_task PROBE-WDIFF && printf cpu )
ok "registered: task.sh -> probe_wdiff.sh, CPU lane, heartbeat probe" '[ "$got" = cpu ] && grep -qF "  PROBE-WDIFF) bash \$RUNBOOK/scripts/probe_wdiff.sh \$G;;" $SRC/scripts/task.sh && grep -qF "\"PROBE-NET2\", \"PROBE-WDIFF\"," $SRC/scripts/heartbeat.sh'
[ $fails = 0 ] && echo "PROBE_WDIFF_TEST PASS ($n checks)" || echo "PROBE_WDIFF_TEST FAIL ($fails of $n)"
