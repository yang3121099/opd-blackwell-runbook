#!/usr/bin/env bash
# v2.5 (D-004 `eval`): one evaluation requested through control/<node>.yaml, on the GPU run_tasks.sh gives it after the
# queue's own tasks. Model sources: Qwen/<name> (the local copy from models.env when it is one of the three models),
# yang31210999/opd-lab-ckpts/runs|ckpts/... (downloaded once to $CKPT_ROOT/adhoc_models/), result:runs/... (this node's
# result repository) or a path under $OPD_ROOT / $CKPT_ROOT. gen_eval runs in resumable blocks (--chunk); results go to
# runs/adhoc_eval_<seq>/ locally and in the result repository; the receipt status/<node>/replies/<seq>.json is updated.
# Usage: bash run_adhoc_eval.sh <gpu> <seq>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
G=$1; S=$2; REQ=$OPD_ROOT/control/requests/${S}_eval.json.taken; OUT=$OPD_ROOT/runs/adhoc_eval_$S; mkdir -p $OUT $OPD_ROOT/control/jobs
echo $$ > $OPD_ROOT/control/jobs/$S.pid   # `cancel <seq>` stops this session
fin() { python $RUNBOOK/scripts/control.py rcpt $S "$1" "$2"; exit 0; }
read -r MODEL BENCH N TEMP <<< "$(python -c "import json,sys; q=json.load(open(sys.argv[1])); print(q['model'], q['bench'], q['n'], q['temp'])" $REQ)" || fin failed "request file unreadable"
case $MODEL in
  Qwen/Qwen3-0.6B) M=$STUDENT_06B;; Qwen/Qwen3-1.7B) M=$STUDENT_17B;; Qwen/Qwen3-8B) M=$TEACHER;; Qwen/*) M=$MODEL;;
  yang31210999/opd-lab-ckpts/*)
    P=${MODEL#yang31210999/opd-lab-ckpts/}; M=$CKPT_ROOT/adhoc_models/$P
    has_weights $M || for ep in ${HF_DOWNLOAD_ENDPOINT:-https://hf-mirror.com} https://huggingface.co; do
      HF_ENDPOINT=$ep python -c "from huggingface_hub import snapshot_download as s; s('yang31210999/opd-lab-ckpts', allow_patterns=['$P/*'], local_dir='$CKPT_ROOT/adhoc_models')" && has_weights $M && break; done;;
  result:*)
    P=${MODEL#result:}; M=$CKPT_ROOT/adhoc_models/result/$P
    PUB_REPO=$(python scripts/remote.py repo public 2>/dev/null); export PUB_REPO   # v2.5.1: weights live in the public repository
    has_weights $M || python -c "
import os; from huggingface_hub import snapshot_download as s
s(os.environ.get('PUB_REPO') or os.environ['HF_REPO'], allow_patterns=['$P/*'], local_dir='$CKPT_ROOT/adhoc_models/result', token=os.environ['HF_TOKEN'], endpoint=os.environ.get('HF_UPLOAD_ENDPOINT', 'https://huggingface.co'))";;
  *) M=$OPD_ROOT/${MODEL#local:}; has_weights $M || M=$CKPT_ROOT/${MODEL#local:};;
esac
has_weights $M || [ "${M#Qwen/}" != "$M" ] || [ -f $M/config.json ] || fin failed "model not found: $MODEL"
case $TEMP in 1.0) SAMP="--temperature 1.0 --top_p 1.0 --top_k 0";; *) SAMP="--temperature 0.6 --top_p 0.95 --top_k 0";; esac
case $BENCH in aime24|aime25|amc23) K=10;; *) K=50;; esac
export CUDA_VISIBLE_DEVICES=$G; cd $RUNBOOK
python code/eval/gen_eval.py --model $M --bench $BENCH --mode no_think --n $N $SAMP --max_tokens 16384 --chunk $K \
  --tag adhoc_${S}_${BENCH} --out $OUT > $OUT/gen_eval.log 2>&1 || fin failed "gen_eval failed: $(tail -1 $OUT/gen_eval.log | cut -c1-200)"
python scripts/remote.py put $OUT runs/adhoc_eval_$S || fin failed "results kept in $OUT; upload failed"
fin ok "$(python -c "import json,glob; m=json.load(open(glob.glob('$OUT/*.metrics.json')[0])); print(f\"{m['bench']} n={m['n']} avg_acc={m['avg_acc']} mean_tokens={m['mean_tokens']} trunc={m['trunc_rate']}\")") -> runs/adhoc_eval_$S"
