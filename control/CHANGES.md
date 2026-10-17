# control/CHANGES.md — sign-off of every change to scripts/control_rules.py

A node running v2.5 or later takes a pulled scripts/control_rules.py only if this file has a line naming its
RULES_VERSION as `rules <version>`; otherwise it resets its clone to the previous commit and reports `rules_refused`
in its heartbeat (R-012). One line per change: date, `rules <version>`, what changed, who approved it.

2026-10-17 rules 2.5.0: first versioned rules (D-004 verbs, process control, default expiry, foreign-process gate); A1 R-012; push approved by Runming
