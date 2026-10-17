# control/CHANGES.md — sign-off of every change to scripts/control_rules.py

A node running v2.5 or later takes a pulled scripts/control_rules.py only if this file has a line naming its
RULES_VERSION as `rules <version>`; otherwise it resets its clone to the previous commit and reports `rules_refused`
in its heartbeat (R-012). One line per change: date, `rules <version>`, what changed, who approved it.

2026-10-17 rules 2.5.0: first versioned rules (D-004 verbs, process control, default expiry, foreign-process gate); A1 R-012; push approved by Runming
2026-10-17 rules 2.5.1: set keys HF_REPO_PUBLIC and HF_REPO_PRIVATE (owner/name; write $OPD_ROOT/hf_repos.txt, uploads split by destination) and HEARTBEAT_SEC (60-3600 s).
2026-10-17 rules 2.5.2: set/setenv key FIXED_MICRO_BSZ (1-16; empty clears it): n sequences per micro-batch instead of the token budget (run_arm.sh).
2026-10-17 rules 2.5.3: setenv key VENV (only fa or empty): run_arm.sh tasks started afterwards use $OPD_ROOT/venv-fa (built and checked by the FA-PROBE task); run accepts FA-PROBE.
2026-10-17 rules 2.5.4: verb reap <gpu> (control.reap: only processes whose parent has died, that are vLLM engines/workers or Ray workers, and whose environment shows this node's task for that GPU; TERM then KILL, an event each). GPU 0-7; refused while a training task (E10b/E12b/SFT/E02/E03) runs on that GPU, since its Ray workers are reaped only after it ended.
2026-10-17 runbook 2.5.4c (rules unchanged 2.5.4): reap also recognises orphans whose process title overwrote /proc/<pid>/environ (vLLM, Ray): ours if it maps a file under OPD_ROOT or its venvs, the GPU from nvidia-smi; env.sh sets SPT_NOENV=1 so new engines keep their environment.
2026-10-17 runbook 2.5.4d (rules unchanged 2.5.4): run_tasks.sh free_gpus counts a GPU as taken while a task.sh or ad-hoc evaluation for it runs, also before the task uses it (FA-PROBE's venv copy left GPU6 empty and EVAL-bridge and EVAL-p5s43-s20 were started on it too).
2026-10-17 runbook 2.5.4e (rules unchanged 2.5.4): models from opd-lab-ckpts come through scripts/fetch_model.py (huggingface.co then hf-mirror.com, longer timeouts, 3 tries each, complete by size and LFS sha256, an event with MB/s and every error); run_tasks.sh prefetches the models of every EVAL task of the queue without a GPU and starts an EVAL task only once its .inputs_ok matches its #eval lines.
