#!/usr/bin/env bash
# [v2.5.8b] Stub tests of probe_e12var.sh kl (no GPU, network or HF): PASS needs the start as the reference (config and
# log), a step-1 KL loss near 0 with a positive one later, and the reward equal to the verifier score.
# Usage: bash scripts/test_probe_e12var.sh -> PROBE_E12VAR_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_e12var.sh $T/rb/scripts/; mkdir -p $T/rb/code/analysis; cp $SRC/code/analysis/e12_rollout_stats.py $T/rb/code/analysis/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_e12.sh <<'SH'
echo "run_e12 $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; S=/runs/E10b-P5/ckpt/global_step_150/actor/huggingface_bf16
if [ $2 = E12b-P5-r3 ]; then   # [v2.5.9w] 128 groups scored a step, FAKE_R3_KEPT kept, their rows in the update and the dump
  echo "${FAKE_R3_HOOK-VARDROP_HOOK_PATCHED}"
  printf "task: E12b-P5-r3\nstart: $S\nalgo: ${FAKE_R3_ALGO:-reinforce_plus_plus_baseline (group mean, batch whitening) 128x8, zero-variance groups dropped}, 4K\nvariant: r3\n" > $d/config.yaml
  python3 - $d <<'PY'
import json, math, os, sys
d = sys.argv[1]; kept = int(os.environ.get('FAKE_R3_KEPT', '20')); var = float(os.environ.get('FAKE_R3_VAR', '1.0'))
extra = int(os.environ.get('FAKE_R3_DUMP_EXTRA', '0')); os.makedirs(os.path.join(d, 'rollouts'), exist_ok=True)
with open(os.path.join(d, 'train.log'), 'a') as lg, open(os.path.join(d, 'vardrop.jsonl'), 'w') as vd:
    for s in (1, 2, 3):
        nz = max(kept, 26)
        for g in range(128):
            vd.write(json.dumps({'step': s, 'group': g, 'rng_seed': 123400000 + s, 'score': [1.0] * 8, 'len': [9] * 8, 'trunc': [0] * 8, 'spread': g < nz, 'kept': g < kept, 'empty_step': False}) + '\n')
        r = 8 * kept
        open(os.path.join(d, 'rollouts', f'{s}.jsonl'), 'w').write('{"score": 1.0}\n' * (r + extra))
        lg.write(f'(TaskRunner pid=1) step:{s} - r3/groups:128.000 - r3/nonzero_groups:{nz}.000 - r3/kept_groups:{kept}.000 - r3/kept_rows:{r}.000 - r3/gen_tokens:900000.000 - r3/kept_tokens:200000.000 - r3/zero_var_frac:{1 - nz / 128:.3f} - r3/empty:0.000 - r3/rng_seed:123400001.000 - r3/adv_mean:0.000 - r3/adv_var:{var:.3f} - r3/update_rows:{r}.000 - r3/update_prompts:{kept}.000 - r3/ppo_mini_batches:{math.ceil(r / 128)}.000 - critic/score/mean:0.5\n')
PY
  exit 0
fi
if [[ $2 == E12c-P5-8k || $2 == E12c-P5-16k || $2 == E12e-P5self-8k ]]; then   # [v2.5.9z85] (Q-077) the 0.6B window arms; [v2.5.9z91] (Q-081) the 8B one
  N=${2##*-}; N=$(( ${N%k} * 1024 ))
  printf "task: $2\nstart: /runs/E10c-P5-fix4/ckpt/global_step_150/actor/huggingface_bf16\nvariant: ${2##*-}\nmax_response_length: ${FAKE_MAXR:-$N}\n" > $d/config.yaml
  for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - response_length/mean:$((5000 + i)) - critic/score/mean:0.4" >> $d/train.log; done
  exit 0
fi
if [ $2 = E12b-P5-16k ]; then   # [v2.5.8h]
  printf "task: E12b-P5-16k\nstart: $S\nvariant: 16k\nmax_response_length: ${FAKE_MAXR:-16384}\n" > $d/config.yaml
  for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - response_length/mean:$((9000 + i)) - critic/score/mean:0.5" >> $d/train.log; done
  exit 0
fi
if [ $2 = E12b-P5-stop2 ]; then   # [v2.5.9f]
  printf "task: E12b-P5-stop2\nstart: $S\nvariant: stop2\nvenv: venv\nreward: ${FAKE_REWARD:-reward_stop2.py (correct x (1 + 0.5 x ...))}\n" > $d/config.yaml
  for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.9 - critic/score/max:1.49 - critic/score/min:0.0 - critic/rewards/mean:0.9 - critic/rewards/max:1.49" >> $d/train.log; done
  mkdir -p $d/rollouts; for g in a a b b; do if [ "${FAKE_NOTAIL:-0}" = 1 ]; then echo '{"input": "'$g'", "output": "o", "score": 1.2}'; else echo '{"input": "'$g'", "output": "o", "score": 1.2, "acc": 1.0, "tail": 400, "cut": 0, "bonus": 0.2}'; fi; done > $d/rollouts/1.jsonl
  exit 0
fi
if [ $2 = E12b-P5-stop ]; then   # [v2.5.8c]
  printf "task: E12b-P5-stop\nstart: $S\nalgo: GRPO, KL 0\nvariant: stop\nvenv: ${FAKE_VENV:-venv}\nreward: ${FAKE_REWARD:-reward_stop.py (1 if correct, +0.5 ...)}\n" > $d/config.yaml
  for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.6 - critic/score/max:${FAKE_MAX:-1.5} - critic/score/min:0.0 - critic/rewards/mean:0.6 - critic/rewards/max:1.5" >> $d/train.log; done
  exit 0
fi
printf "task: E12b-P5-kl\nstart: $S\nalgo: GRPO, ${FAKE_ALGO:-KL loss 0.01 low_var_kl to the start (reference = $S)}\n" > $d/config.yaml
echo "actor_rollout_ref.model.path=$S ref.model.path=${FAKE_REF:-$S}" >> $d/train.log
set -- ${FAKE_KL:-0.0001 0.002 0.004}; i=0
for k in "$@"; do i=$((i + 1)); r=0.5; [ "${FAKE_RW_STEP:-0}" = $i ] && r=0.7
  echo "(TaskRunner pid=1) step:$i - actor/kl_loss:$k - critic/score/mean:0.5 - critic/rewards/mean:$r - timing_s/step:300" >> $d/train.log; done
exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  printf '#!/usr/bin/env bash\necho "tok=$OPD_REWARD_TOKENIZER max=$OPD_REWARD_MAXLEN $*" >> $CALLS; cat > /dev/null; echo "${FAKE_CASES:-[1.5, 1.0, 1.0, 0.0]}"\n' > $T/bin/fakepy; chmod +x $T/bin/fakepy
  export PEV_PY=$T/bin/fakepy
  export CALLS=$T/calls PATH=$T/bin:$PATH PEV_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-KL/report.json "$1"; }
# 1. PASS: reference = the start, step-1 KL ~0 and positive later, reward = score
setup
bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-E12B-KL/.done ]'
ok "3-step smoke of E12b-P5-kl" 'grep -q "^run_e12 2 E12b-P5-kl SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-KL-smoke" $CALLS'
ok "report: KL by step, the reference, reward vs score; upload and event" '[[ "$(res kl_loss_by_step)" == *"0.0001"* ]] && [[ "$(res reward_vs_score)" == *"0.5"* ]] && grep -q "^remote put $T/root/runs/PROBE-E12B-KL/up logs/abcd1234/probe_e12var/PROBE-E12B-KL" $CALLS && grep -q "^control event PROBE-E12B-KL probe_e12var_pass" $CALLS'
rm -rf $T
# 2. the reference is a teacher: a large step-1 KL loss is FAIL
setup
FAKE_KL="0.41 0.40 0.39" bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "step-1 KL too large is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"the reference is not the start"* ]]'
rm -rf $T
# 3. a reward that is not the verifier score is FAIL
setup
FAKE_RW_STEP=2 bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "reward != score is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"reward differs from the verifier score at steps 2"* ]]'
rm -rf $T
# 4. no positive KL loss after step 1 (the loss is not on) is FAIL
setup
FAKE_KL="0 0 0" bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "no KL after step 1 is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"no positive KL loss after step 1"* ]]'
rm -rf $T
# 5. the config does not name the start as the reference, and the log has the start once: FAIL
setup
FAKE_ALGO="KL 0" FAKE_REF=/models/Qwen3-8B bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "reference not the start in config / log is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"does not name the start"* && "$(res reason)" == *"appears 1x"* ]]'
rm -rf $T
# 6. an unknown variant is refused
setup
bash $T/rb/scripts/probe_e12var.sh 2 foo > /dev/null 2>&1; rc=$?
ok "unknown variant exits 2" '[ $rc = 2 ] && [ ! -e $T/root/runs/PROBE-E12B-FOO ]'
rm -rf $T
# [v2.5.8c] stop: 7. PASS (the fixed cases with the start's tokenizer, reward_stop.py in the config, scores in [0, 1.5])
sres() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-STOP/report.json "$1"; }
setup
bash $T/rb/scripts/probe_e12var.sh 2 stop > /dev/null 2>&1
ok "stop PASS" '[ "$(sres result)" = PASS ] && grep -q "^run_e12 2 E12b-P5-stop SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-STOP-smoke" $CALLS && grep -q "^control event PROBE-E12B-STOP probe_e12var_pass" $CALLS'
ok "stop: the fixed cases ran with the start's tokenizer and the 4K limit, on the training's code" 'grep -q "^tok=/runs/E10b-P5/ckpt/global_step_150/actor/huggingface_bf16 max=4096 - $T/rb/code/train" $CALLS && [ "$(sres stop_cases)" = "[1.5, 1.0, 1.0, 0.0]" ]'
rm -rf $T
# 7b. without PEV_PY the fixed cases run on the python of the venv the smoke used (here venv-fa)
setup
mkdir -p $T/root/venv-fa/bin; cp $T/bin/fakepy $T/root/venv-fa/bin/python; sed -i.bak 's/^echo "tok=/echo "venvfa tok=/' $T/root/venv-fa/bin/python
env -u PEV_PY FAKE_VENV=venv-fa bash $T/rb/scripts/probe_e12var.sh 2 stop > /dev/null 2>&1
ok "stop: the fixed cases use the smoke's own venv" '[ "$(sres result)" = PASS ] && grep -q "^venvfa tok=" $CALLS'
rm -rf $T
# 8. the fixed cases do not score as registered (e.g. no bonus anywhere): FAIL
setup
FAKE_CASES="[1.0, 1.0, 1.0, 0.0]" bash $T/rb/scripts/probe_e12var.sh 2 stop > /dev/null 2>&1
ok "stop: wrong fixed-case scores are FAIL" '[ "$(sres result)" = FAIL ] && [[ "$(sres reason)" == *"expected [1.5, 1.0, 1.0, 0.0]"* ]]'
rm -rf $T
# 9. the smoke ran the plain reward: FAIL
setup
FAKE_REWARD=reward_mathverify.py bash $T/rb/scripts/probe_e12var.sh 2 stop > /dev/null 2>&1
ok "stop: the plain reward in the config is FAIL" '[ "$(sres result)" = FAIL ] && [[ "$(sres reason)" == *"does not name reward_stop.py"* ]]'
rm -rf $T
# 10. a score above 1.5: FAIL
setup
FAKE_MAX=2.0 bash $T/rb/scripts/probe_e12var.sh 2 stop > /dev/null 2>&1
ok "stop: a score outside [0, 1.5] is FAIL" '[ "$(sres result)" = FAIL ] && [[ "$(sres reason)" == *"score outside [0, 1.5]"* ]]'
rm -rf $T
# 11. [v2.5.8h] 16k: PASS at the limit, response lengths reported; a config without the limit: FAIL
lres() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-16K/report.json "$1"; }
setup
bash $T/rb/scripts/probe_e12var.sh 2 16k > /dev/null 2>&1
ok "16k PASS with the lengths" '[ "$(lres result)" = PASS ] && [[ "$(lres reason)" == *"at the 16384 limit"*"9001/9002/9003"* ]] && grep -q "^run_e12 2 E12b-P5-16k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-16K-smoke" $CALLS'
rm -rf $T
setup
FAKE_MAXR=4096 bash $T/rb/scripts/probe_e12var.sh 2 16k > /dev/null 2>&1
ok "16k: a smoke at another limit is FAIL" '[ "$(lres result)" = FAIL ] && [[ "$(lres reason)" == *"max_response_length 16384"* ]]'
rm -rf $T
# [v2.5.9z85] (Q-077) PEV_LADDER=E12c: PROBE-E12C-8K / -16K smoke E12c-P5-8k / -16k (their own directories); an E12c variant other than 8k / 16k is refused
cres() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/$1/report.json "$2"; }
setup
PEV_LADDER=E12c bash $T/rb/scripts/probe_e12var.sh 2 8k > /dev/null 2>&1; PEV_LADDER=E12c bash $T/rb/scripts/probe_e12var.sh 2 16k > /dev/null 2>&1
PEV_LADDER=E12c bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1; rkl=$?
ok "[v2.5.9z85] E12c 8k / 16k PASS at their limits in PROBE-E12C-8K / -16K, smoking E12c-P5-<v>; E12c kl refused" '[ "$(cres PROBE-E12C-8K result)" = PASS ] && [[ "$(cres PROBE-E12C-8K reason)" == *"at the 8192 limit"*"5001/5002/5003"* ]] && [ "$(cres PROBE-E12C-16K result)" = PASS ] && [[ "$(cres PROBE-E12C-16K reason)" == *"at the 16384 limit"* ]] && grep -q "^run_e12 2 E12c-P5-8k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12C-8K-smoke" $CALLS && grep -q "^run_e12 2 E12c-P5-16k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12C-16K-smoke" $CALLS && [ $rkl = 2 ] && [ ! -e $T/root/runs/PROBE-E12C-KL ] && [ ! -e $T/root/runs/PROBE-E12B-8K ] && grep -q "^control event PROBE-E12C-8K probe_e12var_pass" $CALLS'
rm -rf $T
setup
PEV_LADDER=E12c FAKE_MAXR=4096 bash $T/rb/scripts/probe_e12var.sh 2 8k > /dev/null 2>&1
ok "[v2.5.9z85] E12c 8k: a smoke at another limit is FAIL" '[ "$(cres PROBE-E12C-8K result)" = FAIL ] && [[ "$(cres PROBE-E12C-8K reason)" == *"max_response_length 8192"* ]]'
rm -rf $T
# [v2.5.9z91] (Q-081) PEV_LADDER=E12e: PROBE-E12E-8K smokes E12e-P5self-8k on its four GPUs; the empty check and the memory samples take
# the fullest GPU of the list (nvidia-smi prints one line per GPU), each GPU is reaped; E12e 16k is refused
setup; printf '#!/usr/bin/env bash\nprintf "100\\n200\\n300\\n400\\n"\n' > $T/bin/nvidia-smi
PEV_LADDER=E12e bash $T/rb/scripts/probe_e12var.sh 0,1,2,3 8k > /dev/null 2>&1; PEV_LADDER=E12e bash $T/rb/scripts/probe_e12var.sh 0,1,2,3 16k > /dev/null 2>&1; r16=$?
ok "[v2.5.9z91] E12e 8k PASS on four GPUs, smoking E12e-P5self-8k on 0,1,2,3; each GPU reaped; peak = the fullest GPU; E12e 16k refused" '[ "$(cres PROBE-E12E-8K result)" = PASS ] && [[ "$(cres PROBE-E12E-8K reason)" == *"at the 8192 limit"* ]] && grep -q "^run_e12 0,1,2,3 E12e-P5self-8k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12E-8K-smoke" $CALLS && [ "$(grep -c "^control reap [0-3] PROBE-E12E-8K$" $CALLS)" = 4 ] && ! grep -qv "^[0-9]*,400$" $T/root/runs/PROBE-E12E-8K/mem.csv && [ $r16 = 2 ] && [ ! -d $T/root/runs/PROBE-E12E-16K ]'
rm -rf $T
# [v2.5.9f] stop2: PASS with the fixed cases in range, the reward file named, the rollout dump carrying the tail
s2() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-STOP2/report.json "$1"; }
setup
FAKE_CASES="[1.4995, 1.41, 0.0, 1.0]" bash $T/rb/scripts/probe_e12var.sh 2 stop2 > /dev/null 2>&1
ok "[v2.5.9f] stop2 PASS: fixed cases, reward_stop2.py, tail in the dump, zero-variance share reported" '[ "$(s2 result)" = PASS ] && grep -q "^run_e12 2 E12b-P5-stop2 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-STOP2-smoke" $CALLS && [[ "$(s2 reason)" == *"zero-variance groups 1.0"* ]]'
rm -rf $T
setup
FAKE_CASES="[1.4995, 1.41, 0.0, 1.0]" FAKE_NOTAIL=1 bash $T/rb/scripts/probe_e12var.sh 2 stop2 > /dev/null 2>&1
ok "[v2.5.9f] stop2: a dump without the parts is FAIL" '[ "$(s2 result)" = FAIL ] && [[ "$(s2 reason)" == *"does not carry the reward"* ]]'
rm -rf $T
setup
FAKE_CASES="[1.5, 1.0, 1.0, 0.0]" bash $T/rb/scripts/probe_e12var.sh 2 stop2 > /dev/null 2>&1
ok "[v2.5.9f] stop2: the old 0 / 0.5 bonus on the fixed cases is FAIL" '[ "$(s2 result)" = FAIL ] && [[ "$(s2 reason)" == *"fixed cases"* ]]'
rm -rf $T
setup
FAKE_CASES="[1.4995, 1.41, 0.0, 1.0]" FAKE_REWARD=reward_stop.py bash $T/rb/scripts/probe_e12var.sh 2 stop2 > /dev/null 2>&1
ok "[v2.5.9f] stop2: another reward in the config is FAIL" '[ "$(s2 result)" = FAIL ] && [[ "$(s2 reason)" == *"does not name reward_stop2.py"* ]]'
rm -rf $T
# [v2.5.9w] r3: the unit tests with the training's python (a fake verl ray_trainer.py with both anchors), then the smoke's checks
r3setup() {
  setup; mkdir -p $T/rb/code/patches $T/py/verl/trainer/ppo
  cp $SRC/code/patches/opd_vardrop.py $SRC/code/patches/opd_lensel.py $SRC/code/patches/lensel_patch.py $SRC/code/patches/vardrop_patch.py $SRC/code/patches/test_opd_vardrop.py $T/rb/code/patches/
  : > $T/py/verl/__init__.py; : > $T/py/verl/trainer/__init__.py; : > $T/py/verl/trainer/ppo/__init__.py
  printf 'class RayPPOTrainer:\n    def fit(self):\n        for epoch in range(3):\n            for batch_dict in []:\n                metrics = {}\n                with marked_timer("step", timing_raw):\n                    batch = batch.union(gen_batch_output)\n                    with marked_timer("adv", timing_raw):\n                        batch = compute_advantage(\n                            batch,\n                            config=self.config.algorithm,\n                        )\n' > $T/py/verl/trainer/ppo/ray_trainer.py
  export PEV_PY=python3 PYTHONPATH=$T/py
}
r3res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-R3/report.json "$1"; }
r3setup; bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1
ok "[v2.5.9w] r3 PASS: unit tests on the node's python, 3-step smoke of E12b-P5-r3, kept groups by step, upload + event" '[ "$(r3res result)" = PASS ] && [[ "$(r3res reason)" == *"kept groups by step 20/20/20 of 128"* ]] && grep -q "passed, 0 failed" $T/root/runs/PROBE-E12B-R3/unit_tests.txt && grep -q "^run_e12 2 E12b-P5-r3 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-R3-smoke" $CALLS && grep -q "^remote put $T/root/runs/PROBE-E12B-R3/up logs/abcd1234/probe_e12var/PROBE-E12B-R3" $CALLS && grep -q "^control event PROBE-E12B-R3 probe_e12var_pass" $CALLS && [ -f $T/root/runs/PROBE-E12B-R3/up/vardrop.jsonl ] && [[ "$(r3res r3_by_step)" == *"dump_rows"* ]]'
rm -rf $T
r3fail() { ok "$1" '[ "$(r3res result)" = FAIL ] && [[ "$(r3res reason)" == *"'"$2"'"* ]]'; }
r3setup; FAKE_R3_VAR=2.2 bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: a variance of 2.2 over the kept tokens (the whitening saw dropped rows) is FAIL" "not whitened on the kept rows"; rm -rf $T
r3setup; FAKE_R3_DUMP_EXTRA=8 bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: dropped rows in the rollout dump is FAIL" "the rollout dump has 168 rows"; rm -rf $T
r3setup; FAKE_R3_KEPT=70 bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: 70 kept groups (over 64) is FAIL" "70 kept groups"; rm -rf $T
r3setup; FAKE_R3_ALGO="GRPO 64x8" bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: a GRPO 64x8 config is FAIL" "does not name reinforce_plus_plus_baseline 128x8"; rm -rf $T
r3setup; FAKE_R3_HOOK="" bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: no hook install before the smoke is FAIL" "vardrop_patch.py did not run"; rm -rf $T
r3setup; printf 'x = 1\n' > $T/py/verl/trainer/ppo/ray_trainer.py; bash $T/rb/scripts/probe_e12var.sh 2 r3 > /dev/null 2>&1; r3fail "[v2.5.9w] r3: the unit tests fail on a ray_trainer.py without the anchors (no smoke)" "r3 unit tests"; ok "[v2.5.9w] r3: no smoke after failed unit tests" '! grep -q "^run_e12" $CALLS'; rm -rf $T
unset PYTHONPATH
[ $fails = 0 ] && echo "PROBE_E12VAR_TEST PASS ($n checks)" || echo "PROBE_E12VAR_TEST FAIL ($fails of $n)"
