"""[v2.5.9z54] (Runming 10-22 21:5x Beijing via Fable, ops-lead A1) the GitHub fallback heartbeat against a local bare repository:
the first round pushes hb/<node>.json; nothing again within GH_HB_SEC (600 s) unless an urgent event (then at once, at most
every 60 s); a failed push backs off 60, 120, ... 900 s and a success resets it; the file carries the pulse, receipts and events
redacted and without this machine's names; GH_HB=0 is off; gh_heartbeat.sh runs one round (GH_HB_ONCE)."""
import json
import os
import subprocess
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402


def git(*a, cwd=None):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd)


@pytest.fixture
def node(monkeypatch):
    d = tempfile.mkdtemp(); remote = os.path.join(d, "hb.git")
    git("init", "-q", "--bare", remote)
    for sub in ("control/replies", "control/events", "logs"):
        os.makedirs(os.path.join(d, "root", sub))
    root = os.path.join(d, "root")
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "CDIR", os.path.join(root, "control"))
    monkeypatch.setenv("OPD_ROOT", root); monkeypatch.setenv("GH_HB_REMOTE", remote); monkeypatch.setenv("GH_HB_GIT_TIMEOUT", "30")
    for k in ("GH_HB", "GH_HB_SEC", "GH_HB_URGENT_MIN_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(C, "_forbidden_tokens", lambda: {"secretuser", "secrethost"})
    json.dump({"tasks": {"E1": {"state": "running", "step": 3}}, "gpus": [{"index": 0, "util_pct": 99}]}, open(C._p("pulse.json"), "w"))
    json.dump({"seq": 482, "cmd": "logs heartbeat 60", "status": "done", "finished_utc": "x", "detail": "ok"}, open(C._p("replies", "482.json"), "w"))
    ev("20261022T100000Z_E1_start.json", "start")
    return remote


def ev(name, kind, detail="d"):
    json.dump({"event": kind, "task": "E1", "utc": "x", "detail": detail}, open(C._p("events", name), "w"))


def remote_file(remote):
    out = git("--git-dir", remote, "show", f"main:hb/{C.node_id()}.json")
    return json.loads(out.stdout) if out.returncode == 0 else None


def commits(remote):
    out = git("--git-dir", remote, "log", "--format=%s", "main")
    return out.stdout.split("\n")[:-1] if out.returncode == 0 else []


def test_first_round_pushes_then_every_600s(node):
    t0 = time.time()
    assert C.gh_round(now=t0) == "pushed (time)"
    f = remote_file(node)
    assert f["node"] == C.node_id() and f["pulse"]["tasks"]["E1"]["step"] == 3 and f["replies"][0]["seq"] == 482
    assert f["control"]["applied_seq"] == 0 and f["events"][0]["event"] == "start" and f["runbook_head"] is not None and f["utc"]
    assert C.gh_round(now=t0 + 10) == "idle" and C.gh_round(now=t0 + 599) == "idle" and len(commits(node)) == 1
    assert C.gh_round(now=t0 + 600) == "pushed (time)" and len(commits(node)) == 2


def test_an_urgent_event_pushes_at_once_at_most_every_60s(node):
    t0 = time.time(); C.gh_round(now=t0)
    ev("20261022T100100Z_E1_fetch_failed.json", "fetch_failed")         # not urgent
    assert C.gh_round(now=t0 + 70) == "idle"
    ev("20261022T100200Z_E1_failed.json", "failed")                     # urgent, 30 s after the last push: waits for 60 s
    assert C.gh_round(now=t0 + 30 + 70) in ("pushed (urgent)",)
    ev("20261022T100300Z_PROBE-X_probe_x_fail.json", "probe_x_fail")
    assert C.gh_round(now=t0 + 120) == "idle"                            # 20 s after that push
    assert C.gh_round(now=t0 + 165) == "pushed (urgent)" and len(commits(node)) == 3
    for k in ("disk_alert", "reserve_low", "watchdog_gave_up", "disk_write_fail"):
        assert k in C.GH_URGENT


def test_backoff_on_a_failed_push_and_reset(node, monkeypatch):
    monkeypatch.setenv("GH_HB_REMOTE", "/nonexistent/hb.git")
    t0 = time.time()
    assert C.gh_round(now=t0).startswith("failed (auth, next try in 60 s)")
    assert C.gh_round(now=t0 + 30) == "backoff"                          # a failed try: again after its backoff, not after 600 s
    ev("20261022T100200Z_E1_failed.json", "failed")
    assert C.gh_round(now=t0 + 61).startswith("failed (auth, next try in 120 s)")   # urgent: tried after 60 s
    ev("20261022T100300Z_E1_failed.json", "failed")
    assert C.gh_round(now=t0 + 61 + 70) == "backoff"                     # urgent and 60 s apart, but the backoff (120 s) lasts
    st = C._gh_state(); assert st["fails"] == 2 and "auth" in st["last_error"]
    waits = []
    for i in range(5):
        r = C.gh_round(now=st["next_try"] + 1); st = C._gh_state(); waits.append(st["backoff_s"])
    assert waits == [240, 480, 900, 900, 900]
    monkeypatch.setenv("GH_HB_REMOTE", node)
    assert C.gh_round(now=st["next_try"] + 1).startswith("pushed") and C._gh_state()["backoff_s"] == 0 and C._gh_state()["fails"] == 0
    assert remote_file(node)["gh_fails_before"] == 7 and remote_file(node)["push_n"] == 1


def test_redaction_and_no_machine_names(node):
    json.dump({"seq": 483, "cmd": "upload secrethost/x", "status": "done", "detail": "read /a/secretuser/b"}, open(C._p("replies", "483.json"), "w"))
    ev("20261022T100500Z_E1_x.json", "x", detail="on secrethost now")
    C.gh_round(now=time.time())
    f = json.dumps(remote_file(node))
    assert "secrethost" not in f and "secretuser" not in f and "<withheld>" in f


def test_off_and_the_setenv_keys(node, monkeypatch):
    open(os.path.join(C.ROOT, "control.env"), "w").write("export GH_HB=0\n")
    assert C.gh_round() == "off" and commits(node) == []
    open(os.path.join(C.ROOT, "control.env"), "w").write("export GH_HB_SEC=60\n")
    t0 = time.time(); C.gh_round(now=t0)
    assert C.gh_round(now=t0 + 60) == "pushed (time)"
    for k, v in (("GH_HB", "0"), ("GH_HB_SEC", "600"), ("GH_HB_URGENT_MIN_S", "60")):
        R.parse({"cmd": f"setenv {k}={v}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 78)


def test_someone_else_pushed_ours_goes_on_top(node):
    t0 = time.time(); C.gh_round(now=t0)
    w = tempfile.mkdtemp(); git("clone", "-q", "-b", "main", node, w); open(os.path.join(w, "README.md"), "w").write("x")
    git("add", "README.md", cwd=w); git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "readme", cwd=w); git("push", "-q", "origin", "HEAD:main", cwd=w)
    assert C.gh_round(now=t0 + 600).startswith("failed (push")
    assert C.gh_round(now=t0 + 600 + 61) == "pushed (time)" and "readme" in commits(node)


def test_ls_remote_self_check_before_the_first_push(node, monkeypatch):   # (Fable 14:4x) `git ls-remote`; a failure: one event
    monkeypatch.setenv("GH_HB_REMOTE", "/nonexistent/hb.git"); evs = []
    monkeypatch.setattr(C, "event", lambda task, kind, detail="": evs.append(kind))
    t0 = time.time()
    assert C.gh_round(now=t0).startswith("failed (auth") and evs == ["gh_hb_auth_failed"]
    assert C.gh_round(now=t0 + 61).startswith("failed (auth") and evs == ["gh_hb_auth_failed"]   # one event per episode
    monkeypatch.setenv("GH_HB_REMOTE", node)
    assert C.gh_round(now=t0 + 61 + 121).startswith("pushed") and C._gh_state()["auth_ok"] is True


def test_https_url_and_the_credential_store(node, monkeypatch, tmp_path):   # the token file is git's, never read here
    cred = tmp_path / "cred"; cred.write_text("https://x:t@github.com\n")
    monkeypatch.setenv("GH_HB_CRED", str(cred)); monkeypatch.delenv("GH_HB_REMOTE")
    seen = []
    real = C.subprocess.run
    def run(cmd, **k):
        seen.append(cmd); return real(["false"], **k) if cmd[:1] == ["git"] else real(cmd, **k)
    monkeypatch.setattr(C.subprocess, "run", run)
    assert C.gh_round(now=time.time()).startswith("failed (auth")
    assert seen[0][:6] == ["git", "-c", "credential.helper=", "-c", f"credential.helper=store --file={cred}", "ls-remote"] and seen[0][6] == C.GH_URL
    assert C.GH_URL == "https://github.com/yang3121099/opd-b300-heartbeat.git"
    src = open(os.path.join(HERE, "control.py")).read()
    i = src.index("def gh_round"); j = src.index("def status():")
    assert "open(cred" not in src[i:j] and "read_text" not in src[i:j]   # this code passes the path to git, never opens it
    open(os.path.join(C.ROOT, "control.env"), "w").write(f"export GH_HB_URL={node}\n")
    monkeypatch.setattr(C.subprocess, "run", real)
    assert C.gh_round(now=time.time() + 4000).startswith("pushed")   # GH_HB_URL from control.env wins


def test_leak_check_over_the_whole_file(node, monkeypatch):   # [ops-lead's A1 condition] defence in depth
    evs = []; monkeypatch.setattr(C, "event", lambda task, kind, detail="": evs.append(kind))
    json.dump({"tasks": {"E1": {"state": "running", "note": "at /data/x"}}, "host": "box.example.com"}, open(C._p("pulse.json"), "w"))
    t0 = time.time(); C.gh_round(now=t0)
    f = remote_file(node)
    assert f["pulse"]["tasks"]["E1"]["note"] == "<withheld>" and f["pulse"]["host"] == "<withheld>" and f["pulse"]["tasks"]["E1"]["state"] == "running"
    assert evs == ["gh_hb_withheld"]
    C.gh_round(now=t0 + 600); assert evs == ["gh_hb_withheld"]           # at most hourly
    C.gh_round(now=t0 + 3700); assert evs == ["gh_hb_withheld", "gh_hb_withheld"]
    assert C.leak_check(f) == []


def test_status_file_age_and_builder_errors(node):
    sf = os.path.join(C.ROOT, "logs", "status_box1.json"); json.dump({"utc": "2026-10-22T11:17:09Z"}, open(sf, "w"))
    os.utime(sf, (time.time() - 3600, time.time() - 3600))
    open(os.path.join(C.ROOT, "logs", "status_builder.err"), "w").write("2026-10-22T15:00:00Z\nrc=1\nTraceback\nNameError: name x is not defined\n")
    C.gh_round(now=time.time()); f = remote_file(node)["status_file"]
    assert 3590 < f["age_s"] < 3700 and f["utc_inside"] == "2026-10-22T11:17:09Z" and f["builder_err"][-1] == "NameError: name x is not defined"


def test_the_loop_script_one_round(node, tmp_path):
    rb = tmp_path / "rb"; (rb / "scripts").mkdir(parents=True)
    for f in ("gh_heartbeat.sh", "control.py", "control_rules.py", "remote.py"):
        (rb / "scripts" / f).write_text(open(os.path.join(HERE, f)).read())
    (rb / "scripts" / "env.sh").write_text(f"export OPD_ROOT={C.ROOT} RUNBOOK={rb}\n")
    r = subprocess.run(["bash", str(rb / "scripts" / "gh_heartbeat.sh")], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, GH_HB_ONCE="1", PATH=os.environ["PATH"]))
    assert "GH_HEARTBEAT_START" in r.stdout and "pushed (time)" in r.stdout, r.stdout + r.stderr
    assert remote_file(node) is not None and os.path.exists(C._p("loop_gh_heartbeat"))
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    assert "ensure_loop gh_heartbeat" in src
