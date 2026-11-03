#!/bin/bash
# [v2.5.9z134] loops.sh: PYTHONDONTWRITEBYTECODE=1 only when node.env links an old base's venv (LINK_VENV=1 and FROM_OPD_ROOT set)
set -u
L=$(grep -n 'export PYTHONDONTWRITEBYTECODE=1' "$(dirname "$0")/loops.sh" | head -1 | cut -d: -f2- | sed 's/   # .*//')
[ -n "$L" ] || { echo "LOOPS_PYC_TEST FAIL: line missing"; exit 1; }
fail=0
chk() {   # LINK_VENV FROM_OPD_ROOT expected
  got=$(env -u PYTHONDONTWRITEBYTECODE LINK_VENV="$1" FROM_OPD_ROOT="$2" bash -c "$L; echo \${PYTHONDONTWRITEBYTECODE:-unset}")
  [ "$got" = "$3" ] || { echo "FAIL LINK_VENV=$1 FROM_OPD_ROOT=$2: $got, want $3"; fail=1; }
}
chk 1 /old/opd-run 1
chk 0 /old/opd-run unset
chk 1 "" unset
chk "" "" unset
[ $fail = 0 ] && echo "LOOPS_PYC_TEST PASS (4 checks)" || exit 1
