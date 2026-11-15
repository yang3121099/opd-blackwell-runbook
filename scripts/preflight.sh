#!/usr/bin/env bash
# preflight.sh (runbook [z168], Fable board v20 rule 9): everything a start of <task> on <gpus> needs on THIS root, checked before it is
# started - through task.sh's own code path (dryrun_json.py runs the real task.sh in a sandbox; only python / torchrun / nvidia-smi are
# recording shims, so it stops exactly where the real start would launch a process): the GPU count (gpus_lib.sh, task.sh's check), the
# queue deps, every gate the runner wants PASS here and every input (.fetch_ok), the node's own first refusal (its control.env and gate
# reports), and for a training where each save goes / how big / the free space there and where a reclaimed run resumes from.
# Exit 0 = ready; 1 = something missing (the lines say what); 2 = no conclusion (the dry run failed). Do not pipe it when it gates a start.
# Usage (on the node, env.sh sourced or OPD_ROOT set): bash scripts/preflight.sh <gpus> <task> && bash scripts/task.sh <gpus> <task>
set -uo pipefail
G=${1:?gpus}; T=${2:?task}; HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "${OPD_ROOT:-}" ] || source "$HERE/env.sh"
exec python3 "$HERE/deps.py" "$T" --root "$OPD_ROOT" --gpus "$G" --preflight
