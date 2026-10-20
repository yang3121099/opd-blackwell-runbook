"""Unit tests of the v2.5 control plane (D-004 / R-012): rules, expiry, receipts, cancel, rate limits, the signal
target rules and the rules sign-off on pull. CPU only, no GPU, network or credentials; nothing is signalled (os.kill
is replaced). Real-process tests: test_control_v25_linux.sh (Linux, a machine without training).
Run: python -m pytest -q scripts/test_control_v25.py"""
import json
import os
import subprocess
import time
import types

import pytest

from test_control import env, write_cmds, results  # noqa: F401  (shared fixture)
import control_rules as R


def iso(dt):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + dt))


def write(e, entries, node="testnode"):
    lines = [f"node: {node}", "commands:"] + ["  - " + json.dumps(x) for x in entries]
    (e.rb / "control" / "testnode.yaml").write_text("\n".join(lines) + "\n")


def rcpt(e, seq):
    p = e.remote / "status" / "testnode" / "replies" / f"{seq}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------- rules ----------
@pytest.mark.parametrize("cmd", ["ps", "snapshot", "snapshot nvidia-smi", "logs E10b-P2 200", "run E10b-P2", "update",
                                 "eval Qwen/Qwen3-1.7B olympiadbench 4 0.6", "eval yang31210999/opd-lab-ckpts/runs/E12-P5/global_step_100/actor/huggingface math500",
                                 "eval result:runs/E10b-P5/global_step_150/actor/huggingface minervamath", "eval runs/E10b-P5/ckpt/global_step_150/actor/huggingface_bf16 aime24 16 1.0",
                                 "set SAVE=150", "set UPLOAD_WEIGHTS=150", "signal 1234 TERM", "signal 1234 KILL force", "kill-pgid 4321",
                                 "kill-pgid 4321 force", "cancel 7", "upload runs/E11b-P2", "pause E10b-P2"])
def test_rules_accept(cmd):
    R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd", ["ps now", "logs E10b-P2 5000", "eval /etc/passwd math500", "eval ../x math500",
                                 "eval Qwen/Qwen3-1.7B gsm8k", "eval Qwen/Qwen3-1.7B math500 17", "eval Qwen/Qwen3-1.7B math500 4 0.7",
                                 "set PATH=/tmp", "set HF_TOKEN=x", "signal 1 TERM", "signal 99 HUP", "signal abc TERM", "kill-pgid 1",
                                 "cancel x", "rm -rf /", "logs E10b;P2 5", "run E10b-P2 now", "update now"])
def test_rules_refuse(cmd):
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd, "reason": "t"})


def test_reason_required():
    with pytest.raises(R.Reject, match="reason"):
        R.parse({"cmd": "ps"})


def test_expiry_defaults():
    now = time.time()
    assert R.expiry({"utc": iso(-3 * 3600)}, "kill") < now          # a 3 h old kill is stale
    assert R.expiry({"utc": iso(-3 * 3600)}, "pause") > now         # a 3 h old pause is not (24 h)
    assert R.expiry({}, "signal") == 0 and R.expiry({}, "pause") == float("inf")
    assert R.expiry({"utc": iso(0), "expires": iso(-60)}, "ps") < now


# ---------- apply: receipts, expiry, cancel, rate limits ----------
def test_receipts_and_status_names(env):
    write(env, [{"seq": 1, "cmd": "pause T", "reason": "r1", "utc": iso(0)},
                {"seq": 2, "cmd": "set PATH=/x", "reason": "r2", "utc": iso(0)},
                {"seq": 3, "cmd": "kill T", "reason": "r3", "utc": iso(-3 * 3600)},
                {"seq": 4, "cmd": "ps", "reason": ""}])
    env.c.apply()
    a, b, c, d = (rcpt(env, i) for i in (1, 2, 3, 4))
    assert a["status"] == "done" and a["reason"] == "r1" and a["node"] == "testnode" and a["rules_version"] == R.RULES_VERSION
    assert b["status"] == "refused" and "PATH" in b["detail"]
    assert c["status"] == "expired"                                   # simulated 3 h offline: the kill never runs
    assert not (env.root / "runs" / "T" / ".killed").exists()
    assert d["status"] == "refused" and "reason" in d["detail"]
    s = env.c.status()
    assert s["control_version"] == R.RULES_VERSION and s["applied_seq"] == 4 and s["pending"] == 0 and s["last_apply_utc"]


def test_applied_seq_before_receipt_upload(env, monkeypatch):
    """v2.5.7zb: a command's applied_seq is recorded before its receipt is uploaded, so a hung or timed-out upload
    (2026-10-19 12:17-13:50 UTC) can never make the next pull run the command again; the receipt is local first."""
    seen = []
    def put(src, dst):
        if "/replies/" in dst:
            seen.append((dst, int(open(os.path.join(env.c.CDIR, "applied_seq")).read()), os.path.exists(src)))
            return False   # as a timed-out upload returns
        return True
    monkeypatch.setattr(env.c, "remote_put", put)
    write(env, [{"seq": 1, "cmd": "pause T", "reason": "r", "utc": iso(0)}, {"seq": 2, "cmd": "pause U", "reason": "r", "utc": iso(0)}])
    env.c.apply()
    assert [x[1] for x in seen] == [1, 2] and all(x[2] for x in seen)   # applied_seq already advanced, receipt on disk
    env.c.apply()                                                        # nothing runs twice
    assert len(seen) == 2 and json.load(open(os.path.join(env.c.CDIR, "replies", "1.json")))["status"] == "done"


def test_remote_put_timeout(env, monkeypatch):
    """v2.5.7zb: remote_put returns False (and logs) when the upload outlives REMOTE_PUT_TIMEOUT."""
    import subprocess as sp
    def hang(*a, **k):
        raise sp.TimeoutExpired(a[0], k.get("timeout"))
    monkeypatch.setattr(env.c.subprocess, "run", hang)
    monkeypatch.setenv("REMOTE_PUT_TIMEOUT", "1")
    os.makedirs(os.path.join(env.c.ROOT, "logs"), exist_ok=True)
    assert env.c.remote_put("/x", "status/n/replies/1.json") is False
    assert "remote_put timeout status/n/replies/1.json" in open(os.path.join(env.c.ROOT, "logs", "control_timeouts.log")).read()


def test_cancel_before_run(env):
    write(env, [{"seq": 1, "cmd": "cancel 2", "reason": "x", "utc": iso(0)}, {"seq": 2, "cmd": "pause T", "reason": "x", "utc": iso(0)}])
    env.c.apply()
    assert rcpt(env, 2)["status"] == "cancelled" and not (env.root / "runs" / "T" / ".paused").exists()


def test_rate_limits(env):
    write(env, [{"seq": i, "cmd": f"signal {1000 + i} TERM", "reason": "x", "utc": iso(0)} for i in range(1, 13)]
          + [{"seq": 13, "cmd": "kill-pgid 5000", "reason": "x", "utc": iso(0)}])
    env.c.apply()
    assert int(open(os.path.join(env.c.CDIR, "applied_seq")).read()) == 10        # 10 signals per pull, the rest waits
    assert env.c.status()["pending"] == 3
    env.c.apply(); env.c.apply()
    assert int(open(os.path.join(env.c.CDIR, "applied_seq")).read()) == 13


def test_killpgid_once_per_pull_without_force(env):
    write(env, [{"seq": 1, "cmd": "kill-pgid 5000", "reason": "x", "utc": iso(0)}, {"seq": 2, "cmd": "kill-pgid 5001", "reason": "x", "utc": iso(0)},
                {"seq": 3, "cmd": "kill-pgid 5002 force", "reason": "x", "utc": iso(0)}])
    env.c.apply()
    assert int(open(os.path.join(env.c.CDIR, "applied_seq")).read()) == 1
    env.c.apply()
    assert int(open(os.path.join(env.c.CDIR, "applied_seq")).read()) == 3        # seq 3 has force: no limit


# ---------- signal target rules (fixture process table; os.kill replaced) ----------
ME = os.getuid()


def proc(pid, cmd, uid=ME, start="2026-10-17T07:00:00Z", h=None):
    return {"pid": pid, "ppid": 1, "pgid": pid, "sid": pid, "uid": uid, "state": "S", "start_utc": start, "cpu_pct": 0, "rss_mb": 1,
            "argv_hash": h or f"h{pid}", "_cmd": cmd}


@pytest.fixture
def table(env, monkeypatch):
    live = {101: proc(101, "python code/eval/gen_eval.py --model m"), 102: proc(102, "python -m verl.trainer.main_ppo x"),
            103: proc(103, "/usr/sbin/sshd -D"), 104: proc(104, "python other.py", uid=ME + 1),
            105: proc(105, "python someone_elses_job.py"), 106: proc(106, "sleep 100"), 107: proc(107, "python code/eval/gen_eval.py", h="NEW")}
    rows = [dict({k: v for k, v in live[p].items() if k != "_cmd"}, cmd=live[p]["_cmd"], ours=ours, gpu=gpu, task=task)
            for p, ours, gpu, task in [(101, True, 0, "E11b-P2"), (102, True, 1, "E10b-P5"), (103, True, None, None), (104, True, 2, None),
                                       (105, False, 3, None), (106, False, None, None), (107, True, 0, "E11b-P3")]]
    rows[-1]["argv_hash"] = "OLD"   # published before the pid was recycled
    json.dump({"rows": rows}, open(os.path.join(env.c.CDIR, "procs.json"), "w"))
    monkeypatch.setattr(env.c, "_proc", lambda p: live.get(p))
    monkeypatch.setattr(env.c, "_ancestors", lambda: {999})
    sent = []
    monkeypatch.setattr(env.c.os, "kill", lambda p, s: sent.append((p, s)))
    monkeypatch.setattr(env.c, "publish_procs", lambda upload=True: {"rows": rows})
    monkeypatch.setattr(env.c.time, "sleep", lambda s: None)
    return env, sent


@pytest.mark.parametrize("pid,sig,flags,why", [
    (555, "TERM", {}, "not in the latest"), (107, "TERM", {}, "recycled"), (103, "TERM", {}, "protected"),
    (104, "TERM", {}, "another user"), (106, "TERM", {}, "not in our process trees"),
    (105, "TERM", {}, "force and ctl.py --confirm-foreign"), (105, "TERM", {"force": True}, "confirm-foreign"),
    (102, "STOP", {}, "training"), (102, "CONT", {}, "training")])
def test_signal_refusals(table, pid, sig, flags, why):
    env, sent = table
    status, msg, _ = env.c.v_signal([str(pid), sig], 1, flags)
    assert status == "rejected" and why in msg, msg
    assert sent == []


def test_signal_allowed_paths(table, monkeypatch):
    env, sent = table
    assert env.c.v_signal(["101", "STOP"], 1, {})[0] == "ok"               # evaluation may be frozen
    assert env.c.v_signal(["101", "CONT"], 2, {})[0] == "ok"
    assert env.c.v_signal(["105", "TERM"], 3, {"force": True, "confirm_foreign": True})[0] == "ok"   # foreign: both flags
    monkeypatch.setattr(env.c, "_alive", lambda p: False)                 # exits after TERM: no KILL needed
    st, msg, _ = env.c.v_signal(["102", "KILL"], 4, {})
    assert st == "ok" and "TERM" in msg and "KILL" not in msg
    import signal as S
    assert sent[:2] == [(101, S.SIGSTOP), (101, S.SIGCONT)] and (105, S.SIGTERM) in sent and (102, S.SIGTERM) in sent


def test_kill_without_force_terms_first(table, monkeypatch):
    env, sent = table
    monkeypatch.setattr(env.c, "_alive", lambda p: True)                  # ignores TERM: KILL after the 30 s wait
    t = [0.0]
    monkeypatch.setattr(env.c.time, "time", lambda: t.__setitem__(0, t[0] + 5) or t[0])
    st, msg, _ = env.c.v_signal(["101", "KILL"], 1, {})
    import signal as S
    assert st == "ok" and [s for _, s in sent] == [S.SIGTERM, S.SIGKILL]


# ---------- rules sign-off on pull ----------
def test_check_pull_requires_signoff(env):
    rb = str(env.rb); g = lambda *a: subprocess.run(["git", "-C", rb, *a], capture_output=True, text=True, check=True).stdout.strip()
    (env.rb / "scripts" / "control_rules.py").write_text('RULES_VERSION = "2.5.0"\n'); g("add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "a"); a = g("rev-parse", "--short", "HEAD")
    (env.rb / "scripts" / "control_rules.py").write_text('RULES_VERSION = "2.5.1"\nWIDER = True\n'); g("add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"); b = g("rev-parse", "--short", "HEAD")
    assert env.c.check_pull(a, b) is False and g("rev-parse", "--short", "HEAD") == a          # reset to the old rules
    assert "2.5.1" in env.c.status()["rules_refused"]
    g("reset", "-q", "--hard", b)
    (env.rb / "control" / "CHANGES.md").write_text("2026-10-17 rules 2.5.1: wider eval sources (Runming)\n"); g("add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"); c = g("rev-parse", "--short", "HEAD")
    assert env.c.check_pull(a, c) is True


# ---------- ad-hoc requests ----------
def test_requests_in_seq_order_and_cancelled(env, capsys):
    env.c._request("eval", 12, {"model": "Qwen/Qwen3-1.7B", "bench": "math500"}); env.c._request("run", 9, {"task": "E11b-P2"})
    env.c._request("run", 15, {"task": "E11b-P3"}); os.makedirs(os.path.join(env.c.CDIR, "cancelled"), exist_ok=True)
    open(os.path.join(env.c.CDIR, "cancelled", "12"), "w").write("x")
    env.c.next_request(); env.c.next_request(); env.c.next_request()
    assert capsys.readouterr().out.split("\n")[:3] == ["run E11b-P2 9", "run E11b-P3 15", ""]


# ---------- heartbeat on demand and alerts ----------
def test_heartbeat_flag(env):
    assert env.c.run_one("heartbeat", 3)[0] == "ok" and os.path.exists(os.path.join(env.c.CDIR, "heartbeat_now"))


def events(e):
    d = e.remote / "status" / "testnode" / "events"
    return sorted(json.loads(p.read_text())["event"] for p in d.glob("*.json")) if d.exists() else []


def test_alerts_once_and_cleared(env, tmp_path, monkeypatch):
    stp = tmp_path / "st.json"
    monkeypatch.setattr(env.c, "running_pids", lambda task=None: [])
    stp.write_text(json.dumps({"disk_alert": True, "ckpt_disk_alert": False, "disk_free_gb": 12}))
    assert env.c.alerts(str(stp)) == ["disk_alert"]
    assert env.c.alerts(str(stp)) == ["disk_alert"]                       # still on: no second event
    stp.write_text(json.dumps({"disk_alert": False}))
    assert env.c.alerts(str(stp)) == []
    assert events(env) == ["disk_alert", "disk_alert_cleared"] and env.c.status()["alerts"] == []


def test_ckpt_root_alert_symbolic_target(env, tmp_path, monkeypatch):
    """v2.5.8: a checkpoint-root alert is evented as root<i> (never its path: a "/" in the event file name raised
    FileNotFoundError every round from 2026-10-17 on), the state is saved before the events, and one failing event
    does not stop the others."""
    monkeypatch.setattr(env.c, "running_pids", lambda task=None: [])
    stp = tmp_path / "st.json"
    stp.write_text(json.dumps({"disk_alert": True, "ckpt_roots": [{"base": "/a/b", "alert": False, "writable": True, "free_gb": 5000},
                                                                     {"base": "/x/y/z", "alert": True, "writable": True, "free_gb": 72.3}]}))
    assert env.c.alerts(str(stp)) == ["ckpt_root_alert:root1", "disk_alert"]
    ev = [e for e in os.listdir(os.path.join(env.c.CDIR, "events"))]
    assert any(e.endswith("_root1_ckpt_root_alert.json") for e in ev) and not any("x_y_z" in e for e in ev)
    real = env.c.event
    def flaky(task, kind, detail=""):
        if kind == "disk_alert_cleared":
            raise OSError("boom")
        return real(task, kind, detail)
    monkeypatch.setattr(env.c, "event", flaky)
    stp.write_text(json.dumps({"ckpt_roots": []}))
    assert env.c.alerts(str(stp)) == []                                    # the failing event did not stop the other
    assert env.c.status()["alerts"] == [] and "ckpt_root_alert_cleared" in events(env)


def test_stalled_and_missing_uploader(env, tmp_path, monkeypatch):
    logs = env.root / "logs"; logs.mkdir(exist_ok=True); lg = logs / "E10b-P2.log"; lg.write_text("step 3\n")
    old = time.time() - 3600; os.utime(lg, (old, old))
    monkeypatch.setattr(env.c, "running_pids", lambda task=None: [4242])
    real_read = env.c._read
    monkeypatch.setattr(env.c, "_read", lambda p, d="": "bash\0scripts/task.sh\0003\0E10b-P2\0" if p == "/proc/4242/cmdline" else real_read(p, d))
    real_run = env.c.subprocess.run                                       # only pgrep is faked (no upload_sync running)
    monkeypatch.setattr(env.c.subprocess, "run", lambda args, *a, **k: types.SimpleNamespace(stdout="", returncode=0) if args[0] == "pgrep" else real_run(args, *a, **k))
    stp = tmp_path / "st.json"; stp.write_text("{}")
    assert env.c.alerts(str(stp)) == ["stalled:E10b-P2", "upload_sync_missing:E10b-P2"]
    os.utime(lg, None)                                                     # the log moves again
    assert env.c.alerts(str(stp)) == ["upload_sync_missing:E10b-P2"]
    assert "stalled_cleared" in events(env) and "stalled" in events(env)


# ---------- checkpoint roots (v2.5): candidates, switching, move-ckpt, report ----------
@pytest.fixture
def roots(env, tmp_path, monkeypatch):
    b1, b2 = tmp_path / "home", tmp_path / "bigdisk"; b1.mkdir(); b2.mkdir()
    monkeypatch.setattr(env.c, "candidates", lambda: [str(b1), str(b2)])
    return env, b1, b2


def paths_env(e):
    return dict(l.split("=", 1) for l in (e.root / "paths.env").read_text().splitlines() if "=" in l)


def test_set_ckpt_root_only_candidates(roots, tmp_path):
    env, b1, b2 = roots
    assert env.c.run_one("set CKPT_ROOT=/elsewhere", 1)[0] == "rejected"
    assert env.c.run_one(f"set CKPT_ROOT={b2}", 2)[0] == "ok" and paths_env(env)["CKPT_ROOT"] == f"{b2}/opd-ckpt"
    assert env.c.run_one(f"set CKPT_ROOT={b1}/opd-ckpt", 3)[0] == "ok" and paths_env(env)["CKPT_ROOT"] == f"{b1}/opd-ckpt"
    assert (b2 / "opd-ckpt").is_dir()


def test_add_candidate_and_archive(roots, tmp_path):
    env, b1, b2 = roots
    new = tmp_path / "third"; new.mkdir()
    assert env.c.run_one(f"set CKPT_CANDIDATES+={tmp_path}/missing", 1)[0] == "rejected"
    assert env.c.run_one(f"set CKPT_CANDIDATES+={new}", 2)[0] == "ok" and paths_env(env)["CKPT_CANDIDATES_EXTRA"] == str(new)
    assert env.c.run_one(f"set ARCHIVE_ROOT={b2}", 3)[0] == "ok" and paths_env(env)["ARCHIVE_ROOT"] == str(b2)
    assert env.c.run_one("set ARCHIVE_ROOT=", 4)[0] == "ok" and "ARCHIVE_ROOT" not in paths_env(env)


def test_move_ckpt(roots, monkeypatch):
    env, b1, b2 = roots
    src = b1 / "opd-ckpt" / "E10b-P2" / "ckpt" / "global_step_150" / "actor"; src.mkdir(parents=True)
    (src / "w.safetensors").write_bytes(b"x" * 1000); (src.parent.parent / "latest_checkpointed_iteration.txt").write_text("150")
    run = env.root / "runs" / "E10b-P2"; run.mkdir(parents=True); os.symlink(b1 / "opd-ckpt" / "E10b-P2" / "ckpt", run / "ckpt")
    monkeypatch.setattr(env.c, "running_pids", lambda task=None: [1])
    assert "running" in env.c.v_move_ckpt(["E10b-P2", str(b2)], 1)[1]
    monkeypatch.setattr(env.c, "running_pids", lambda task=None: [])
    assert env.c.v_move_ckpt(["E10b-P2", "/not/a/candidate"], 2)[0] == "rejected"
    real_popen = env.c.subprocess.Popen                                  # only the background do_move launch runs inline
    monkeypatch.setattr(env.c.subprocess, "Popen", lambda args, **k: env.c.do_move(int(args[4]), args[5], args[6], args[7]) if args[0] == "setsid" else real_popen(args, **k))
    st, msg, _ = env.c.v_move_ckpt(["E10b-P2", str(b2)], 3)
    dst = b2 / "opd-ckpt" / "E10b-P2" / "ckpt"
    assert st == "accepted" and os.path.realpath(run / "ckpt") == str(dst.resolve())
    assert (dst / "global_step_150" / "actor" / "w.safetensors").read_bytes() == b"x" * 1000
    assert not (b1 / "opd-ckpt" / "E10b-P2" / "ckpt").exists() and (run / ".paused").exists()
    assert rcpt(env, 3)["status"] == "done" and rcpt(env, 3)["bytes"] == 1003


def test_roots_report(roots, monkeypatch):
    env, b1, b2 = roots
    (b2 / "opd-ckpt" / "E12b-P5" / "ckpt").mkdir(parents=True); (b2 / "opd-ckpt" / "E12b-P5" / "ckpt" / "f").write_bytes(b"y" * 2_000_000)
    monkeypatch.setenv("CKPT_ROOT", f"{b2}/opd-ckpt")
    rep = {r["base"]: r for r in env.c.roots_report()}
    assert rep[str(b2)]["current"] and not rep[str(b1)]["current"]
    assert rep[str(b2)]["tasks_gb"] == {"E12b-P5": 0.0} or rep[str(b2)]["tasks_gb"]["E12b-P5"] >= 0
    assert rep[str(b1)]["writable"] and rep[str(b1)]["free_gb"] > 0 and "mount" in rep[str(b1)]


# ---------- [v2.5.8] loop protection by the process's own program (kill-pgid of a run_tasks-launched task group) ----------
WRAP = "bash -c bash /r/scripts/task.sh 3 EVAL-x > /l/EVAL-x.log 2>&1 ; python /r/scripts/control.py event EVAL-x done x"


def test_protected_by_program_not_arguments():
    for cmd, want in ((WRAP, False), ("bash /r/scripts/task.sh 3 EVAL-x", False), ("python3 code/eval/gen_eval.py --note scripts/control.py", False),
                      ("python -u /r/scripts/control.py apply", True), ("python3 /r/scripts/control.py procs --upload", True),
                      ("bash /r/scripts/run_tasks.sh", True), ("bash scripts/heartbeat.sh", True), ("bash /r/scripts/watchdog.sh", True),
                      ("/usr/sbin/sshd -D", True), ("tmux new -s x", True), ("timeout 5400 bash /r/scripts/run_queue_eval.sh prefetch E", False)):
        assert R.protected(cmd) is want, cmd


def killpgid_env(env, monkeypatch, members):
    live = {p: proc(p, c) for p, c in members}
    for p in live:
        live[p]["pgid"] = members[0][0]
    rows = [dict({k: v for k, v in live[p].items() if k != "_cmd"}, cmd=live[p]["_cmd"], ours=True, gpu=None, task="EVAL-x") for p in live]
    json.dump({"rows": rows}, open(os.path.join(env.c.CDIR, "procs.json"), "w"))
    monkeypatch.setattr(env.c, "_proc", lambda p: live.get(p))
    monkeypatch.setattr(env.c, "proc_table", lambda: rows)
    monkeypatch.setattr(env.c, "_ancestors", lambda: {999})
    killed = []
    monkeypatch.setattr(env.c.os, "killpg", lambda g, s: (killed.append((g, s)), live.clear()))
    monkeypatch.setattr(env.c, "publish_procs", lambda upload=True: {"rows": rows})
    monkeypatch.setattr(env.c.time, "sleep", lambda s: None)
    return killed


def test_killpgid_task_group_with_control_py_argument(env, monkeypatch):
    killed = killpgid_env(env, monkeypatch, [(7000, WRAP), (7001, "bash /r/scripts/task.sh 3 EVAL-x"), (7002, "python code/eval/gen_eval.py --m x")])
    st, detail, extra = env.c.v_killpgid(["7000"], 1)
    assert st == "ok" and killed and killed[0][0] == 7000 and extra["members"] == [7000, 7001, 7002], (st, detail)


def test_killpgid_refuses_a_group_holding_a_loop(env, monkeypatch):
    killed = killpgid_env(env, monkeypatch, [(7100, WRAP), (7101, "python -u /r/scripts/control.py apply")])
    st, detail, _ = env.c.v_killpgid(["7100"], 1)
    assert st == "rejected" and "protected" in detail and not killed
