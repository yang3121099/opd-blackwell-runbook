#!/usr/bin/env bash
# [z153] Stub test of upload_sync.sh's final-step rule and the reading race of 11-11 (no HF, no torch, no GPU): the real
# upload_sync.sh runs against stubs (remote.py put = PUBLIC_UPLOADS=0's defer_weight: a hard-linked hold under
# $CKPT_ROOT/weights_pending; to_bf16.py = a copy; sleep a no-op), then run_queue_eval.sh's own local_save (LOCAL_FIRST) is asked
# for the weights, as a reading that starts after upload_sync has passed would be.
# Usage: bash scripts/test_upload_final.sh -> UPLOAD_FINAL_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {   # <config.yaml text>: a finished run E99-X with steps 25 and 50 (fp32 HF weights, older than 120 s)
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/venv/bin $T/ck $T/bin; : > $T/root/venv/bin/activate
  cp $SRC/scripts/upload_sync.sh $T/rb/scripts/
  printf 'export OPD_ROOT=%s RUNBOOK=%s CKPT_ROOT=%s\n' $T/root $T/rb $T/ck > $T/rb/scripts/env.sh
  cat > $T/rb/scripts/remote.py <<'PY'
import os, sys
a = sys.argv[1:]
open(os.environ["CALLS"], "a").write("remote " + " ".join(a) + "\n")
if a[:1] == ["put"] and "/huggingface" in a[2]:   # defer_weight: the files hard-linked into the hold, the source left as it is
    d = os.path.join(os.environ["CKPT_HOLD"], a[2]); os.makedirs(d, exist_ok=True)
    for f in os.listdir(a[1]):
        os.link(os.path.join(a[1], f), os.path.join(d, f))
PY
  printf 'import os, shutil, sys\nshutil.copytree(sys.argv[1], sys.argv[2])\nopen(os.path.join(sys.argv[2], "model.safetensors"), "w").write("bf16")\n' > $T/rb/scripts/to_bf16.py
  printf 'import sys\n' > $T/rb/scripts/metrics_jsonl.py; printf 'import sys\n' > $T/rb/scripts/control.py
  printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python
  printf '#!/usr/bin/env bash\nexit 0\n' > $T/bin/sleep
  printf '#!/usr/bin/env bash\n[ "$1" = -c ] && [ "$2" = %%Y ] && exec python3 -c "import os, sys; print(int(os.path.getmtime(sys.argv[1])))" "$3"\nexec /usr/bin/stat "$@"\n' > $T/bin/stat
  chmod +x $T/bin/*
  R=$T/root/runs/E99-X; mkdir -p $R/ckpt; printf "$1" > $R/config.yaml; echo 50 > $R/ckpt/latest_checkpointed_iteration.txt
  for s in 25 50; do mkdir -p $R/ckpt/global_step_$s/actor/huggingface; echo fp32 > $R/ckpt/global_step_$s/actor/huggingface/model.safetensors
    echo '{}' > $R/ckpt/global_step_$s/actor/huggingface/config.json; touch -t 202611110000 $R/ckpt/global_step_$s/actor/huggingface/*; done
  export CALLS=$T/calls CKPT_HOLD=$T/ck/weights_pending; : > $CALLS
}
run() { env PATH=$T/bin:$PATH "$@" UPLOAD_WEIGHTS=25,50 bash $T/rb/scripts/upload_sync.sh $R > $T/out 2>&1; }
H() { echo $R/ckpt/global_step_$1/actor/$2; }
# 1. config "steps: 50": the final step keeps its fp32, the earlier uploaded step loses it; both keep a bf16 copy (KEEP_BF16 default 1)
setup 'task: E99-X\nsteps: 50\nfinished: now\n'; run
ok "both steps uploaded (held), the loop ended" 'grep -q "UP_OK E99-X step 25" $T/out && grep -q "UP_OK E99-X step 50" $T/out && grep -q "UPLOAD_DONE E99-X" $T/out && [ -f $CKPT_HOLD/runs/E99-X/global_step_50/actor/huggingface/model.safetensors ]'
ok "final step 50: fp32 kept, KEPT_FINAL_HF logged" '[ -f $(H 50 huggingface)/model.safetensors ] && grep -q "^KEPT_FINAL_HF E99-X step 50$" $T/out'
ok "step 25 (not final): fp32 deleted, its huggingface_bf16 kept" '[ ! -f $(H 25 huggingface)/model.safetensors ] && [ -f $(H 25 huggingface_bf16)/model.safetensors ] && ! grep -q "KEPT_FINAL_HF E99-X step 25" $T/out'
ok "the bf16 copy is the held file itself (hard link: no extra disk)" '[ $(H 50 huggingface_bf16)/model.safetensors -ef $CKPT_HOLD/runs/E99-X/global_step_50/actor/huggingface/model.safetensors ]'
# 4. the race: a reading's LOCAL_FIRST lookup (run_queue_eval.sh local_save) after upload_sync's pass still finds weights
has_weights() { ls "$1"/*.safetensors > /dev/null 2>&1; }
eval "$(sed -n '/^local_save() {/,/^  done; }/p' $SRC/scripts/run_queue_eval.sh)"
OPD_ROOT=$T/root CKPT_ROOT=$T/ck
ok "the reading's local_save finds step 50 (bf16 first) and step 25 (bf16)" '[ "$(local_save runs/E99-X/global_step_50/actor/huggingface)" = "$(H 50 huggingface_bf16) auto" ] && [ "$(local_save runs/E99-X/global_step_25/actor/huggingface)" = "$(H 25 huggingface_bf16) auto" ]'
rm -rf $T
# 2. no "steps:" in config.yaml (run_e12.sh before z153): the newest step of a finished run counts as final
setup 'task: E99-X\nfinished: now\n'; run
ok "no steps: line, finished: the newest step (50) keeps its fp32, step 25 does not" '[ -f $(H 50 huggingface)/model.safetensors ] && [ ! -f $(H 25 huggingface)/model.safetensors ] && grep -q "^KEPT_FINAL_HF E99-X step 50$" $T/out'
rm -rf $T
# 3. KEEP_BF16=0 asked for explicitly: no bf16 copy, but the final fp32 stays, so a reading still has weights
setup 'task: E99-X\nsteps: 50\nfinished: now\n'; run KEEP_BF16=0
has_weights() { ls "$1"/*.safetensors > /dev/null 2>&1; }; OPD_ROOT=$T/root CKPT_ROOT=$T/ck
ok "KEEP_BF16=0: no huggingface_bf16, the final fp32 still read by local_save" '[ ! -e $(H 50 huggingface_bf16) ] && [ "$(local_save runs/E99-X/global_step_50/actor/huggingface)" = "$(H 50 huggingface) bfloat16" ]'
ok "KEEP_LOCAL_HF=1 is not needed for that (the default 0 deletes only step 25)" '[ ! -f $(H 25 huggingface)/model.safetensors ]'
rm -rf $T
# 5. the launchers no longer force KEEP_BF16=0; run_e12.sh's config names its steps
ok "run_e12.sh / run_tasks.sh: KEEP_BF16 defaults to 1 when upload_sync starts; config.yaml gets steps:" '! grep -q "KEEP_BF16=0 UPLOAD_WEIGHTS" $SRC/scripts/run_e12.sh $SRC/scripts/run_tasks.sh && grep -q "|| KEEP_BF16=\${KEEP_BF16:-1} UPLOAD_WEIGHTS=" $SRC/scripts/run_e12.sh && [ "$(grep -c "env KEEP_BF16=\${KEEP_BF16:-1} UPLOAD_WEIGHTS=" $SRC/scripts/run_tasks.sh)" = 3 ] && grep -q "^steps: \$STEPS$" $SRC/scripts/run_e12.sh'
[ $fails = 0 ] && echo "UPLOAD_FINAL_TEST PASS ($n checks)" || echo "UPLOAD_FINAL_TEST FAIL ($fails of $n)"
