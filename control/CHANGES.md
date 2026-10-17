# control/CHANGES.md — sign-off of every change to scripts/control_rules.py

A node running v2.5 or later takes a pulled scripts/control_rules.py only if this file has a line naming its
RULES_VERSION as `rules <version>`; otherwise it resets its clone to the previous commit and reports `rules_refused`
in its heartbeat (R-012). One line per change: date, `rules <version>`, what changed, who approved it.

2026-10-17 rules 2.5.0: first versioned rules (D-004 verbs, process control, default expiry, foreign-process gate); A1 R-012; push approved by Runming
2026-10-17 rules 2.5.1: set keys HF_REPO_PUBLIC and HF_REPO_PRIVATE (owner/name; write $OPD_ROOT/hf_repos.txt, uploads split by destination) and HEARTBEAT_SEC (60-3600 s).
2026-10-17 rules 2.5.2: set/setenv key FIXED_MICRO_BSZ (1-16; empty clears it): n sequences per micro-batch instead of the token budget (run_arm.sh).
2026-10-17 rules 2.5.3: setenv key VENV (only fa or empty): run_arm.sh tasks started afterwards use $OPD_ROOT/venv-fa (built and checked by the FA-PROBE task); run accepts FA-PROBE.
