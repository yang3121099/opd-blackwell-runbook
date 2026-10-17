#!/usr/bin/env bash
# Real-process test of the v2.5 process control (D-004 section 5, R-012) on Linux. It starts its own fake processes in a
# temp tree and signals only those (pids it created); still, run it only on a machine without training (PROTOCOL 6).
# Checks: procs.json rows (our tree, task label, argv hash), STOP/CONT on an evaluation (it resumes), STOP refused on
# a training process, TERM on one child, recycled-pid refusal, KILL without force = TERM + 30 s + KILL, kill-pgid of a
# task session. About 1 minute. Usage: bash scripts/test_control_v25_linux.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/eval $T/root $T/remote
cp $SRC/scripts/control.py $SRC/scripts/control_rules.py $T/rb/scripts/
cat > $T/rb/scripts/remote.py <<'PY'
import shutil, sys, os
src, dst = sys.argv[2], os.path.join(os.environ["T_REMOTE"], sys.argv[3])
shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else (os.makedirs(os.path.dirname(dst), exist_ok=True), shutil.copy(src, dst))
PY
cat > $T/rb/scripts/task.sh <<EOF
python3 -c "import time; time.sleep(600)" verl.trainer.main_ppo &
sleep 600 & sleep 600 & wait
EOF
cat > $T/rb/code/eval/gen_eval.py <<'PY'
import sys, time
n = 0
while True:
    n += 1; open(sys.argv[1], "w").write(str(n)); time.sleep(0.2)
PY
cat > $T/stubborn.sh <<'EOF'
trap '' TERM; while true; do sleep 1; done
EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb OPD_NODE=linuxtest T_REMOTE=$T/remote
setsid bash $T/rb/scripts/task.sh 0 FAKE-T > /dev/null 2>&1 < /dev/null &
setsid python3 $T/rb/code/eval/gen_eval.py $T/counter > /dev/null 2>&1 < /dev/null &
( exec -a "bash $T/rb/scripts/task.sh 1 FAKE-S" bash $T/stubborn.sh ) > /dev/null 2>&1 < /dev/null &
sleep 2
python3 - $T <<'PY'
import json, os, sys, time
T = sys.argv[1]; sys.path.insert(0, f"{T}/rb/scripts")
import control as c
os.makedirs(c.CDIR, exist_ok=True)
ok, fails = 0, []
def check(name, cond, info=""):
    global ok
    if cond: ok += 1
    else: fails.append(f"{name}: {info}")
t = c.publish_procs(upload=True); rows = t["rows"]
by = lambda pat: [r for r in rows if pat in r["cmd"]]
task, ev, tr, sl, stub = by("scripts/task.sh 0 FAKE-T"), by("code/eval/gen_eval.py"), by("verl.trainer.main_ppo"), by("sleep 600"), by("FAKE-S")
check("rows", task and ev and tr and len(sl) >= 2 and stub, [r["cmd"][:60] for r in rows])
check("ours and task label", all(r["ours"] for r in task + ev + tr + sl) and tr[0]["task"] == "FAKE-T", tr)
check("argv hash", all(len(r["argv_hash"]) == 16 for r in rows))
check("uploaded", os.path.exists(f"{T}/remote/status/linuxtest/procs.json"))
cnt = lambda: int(open(f"{T}/counter").read() or 0)
st, msg, _ = c.v_signal([str(ev[0]["pid"]), "STOP"], 1, {}); a = cnt(); time.sleep(1.5); b = cnt()
check("STOP eval", st == "ok" and a == b, (st, msg, a, b))
st, msg, _ = c.v_signal([str(ev[0]["pid"]), "CONT"], 2, {}); time.sleep(1.5); d = cnt()
check("CONT eval resumes", st == "ok" and d > b, (st, msg, b, d))
st, msg, _ = c.v_signal([str(tr[0]["pid"]), "STOP"], 3, {})
check("STOP training refused", st == "rejected" and "training" in msg, msg)
st, msg, _ = c.v_signal([str(sl[0]["pid"]), "TERM"], 4, {}); time.sleep(1)
check("TERM one child", st == "ok" and not c._alive(sl[0]["pid"]), msg)
check("sibling untouched", c._proc(sl[1]["pid"]) is not None, "")
t = json.load(open(c._p("procs.json")))
for r in t["rows"]:
    if r["pid"] == sl[1]["pid"]: r["argv_hash"] = "0" * 16
json.dump(t, open(c._p("procs.json"), "w"))
st, msg, _ = c.v_signal([str(sl[1]["pid"]), "TERM"], 5, {})
check("recycled pid refused", st == "rejected" and "recycled" in msg, msg)
c.publish_procs(upload=False)
t0 = time.time(); st, msg, _ = c.v_signal([str(stub[0]["pid"]), "KILL"], 6, {})
check("KILL = TERM + 30 s + KILL", st == "ok" and "TERM+KILL" in msg and time.time() - t0 >= 29, (st, msg, round(time.time() - t0)))
c.publish_procs(upload=False)
st, msg, extra = c.v_killpgid([str(task[0]["pgid"])], 7, {}); time.sleep(1)
left = [m for m in extra.get("members", []) if c._alive(m)]
check("kill-pgid task session", st == "ok" and not left, (st, msg, left))
os.kill(ev[0]["pid"], 9)
print(f"LINUX_PROCESS_TEST {'PASS' if not fails else 'FAIL'}: {ok} checks passed" + ("".join("\n  FAIL " + f for f in fails)))
PY
echo "temp tree: $T (fake processes all stopped)"
