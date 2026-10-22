"""[v2.5.9z50] (Runming 10-22 20:2x Beijing via the reviewer; the node lost the public repository at 11:17 UTC) private first:
every upload to the private repository by default (PUBLIC_UPLOADS=1 restores the split), fetches in the order private, public,
fetch_model.py trying each repository of a comma list, the outbox going on past a failed item, the pulse's new fields."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402
import fetch_model as F  # noqa: E402
import remote as RM  # noqa: E402


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "hf_repos.txt"), "w").write("public=me/pub\nprivate=me/priv\n")
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.delenv("PUBLIC_UPLOADS", raising=False)
    return d


def test_routing_private_first_by_default_and_the_switch(root):
    assert RM.public_paused() and RM.klass("runs/E1/x.json") == "private" and RM.klass("pulse/n.json") == "private"
    assert RM.fetch_repos() == "me/priv,me/pub"
    open(os.path.join(root, "control.env"), "w").write("export PUBLIC_UPLOADS=1\n")
    assert not RM.public_paused() and RM.klass("runs/E1/x.json") == "public" and RM.klass("status/n.json") == "private"
    assert RM.fetch_repos() == "me/pub"


def test_repo_cli(root):
    run = lambda kind: subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "repo", kind], capture_output=True, text=True,
                                      env=dict(os.environ, OPD_ROOT=root)).stdout.strip()
    assert run("public") == "me/priv,me/pub" and run("private") == "me/priv"


def test_fetch_model_tries_each_repository(root, monkeypatch, capsys):
    calls = []
    def lf(repo, prefix, local):
        calls.append(repo)
        return {"ok": repo == "me/pub", "errors": [] if repo == "me/pub" else [{"endpoint": "https://huggingface.co", "attempt": 1, "error": "x"}]}
    monkeypatch.setattr(F, "locked_fetch", lf)
    monkeypatch.setattr(sys, "argv", ["fetch_model.py", "me/priv,me/pub", "runs/E1/global_step_1/actor/huggingface", tempfile.mkdtemp()])
    with pytest.raises(SystemExit) as e:
        F.main()
    r = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert e.value.code == 0 and calls == ["me/priv", "me/pub"] and r["repo"] == "me/pub" and r["repos_tried"] == 2
    calls.clear(); monkeypatch.setattr(F, "locked_fetch", lambda repo, p, l: (calls.append(repo), {"ok": False, "errors": [{"endpoint": repo, "attempt": 1, "error": "x"}]})[1])
    with pytest.raises(SystemExit) as e:
        F.main()
    r = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert e.value.code == 1 and calls == ["me/priv", "me/pub"] and len(r["errors"]) == 2
    monkeypatch.setattr(sys, "argv", ["fetch_model.py", "me/pub", "p", tempfile.mkdtemp()]); calls.clear()
    monkeypatch.setattr(F, "locked_fetch", lambda repo, p, l: (calls.append(repo), {"ok": True})[1])
    with pytest.raises(SystemExit):
        F.main()
    assert calls == ["me/pub"]   # one repository: as before


def test_outbox_goes_on_past_a_failed_item(root, monkeypatch):
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "CDIR", os.path.join(root, "control"))
    os.makedirs(os.path.join(root, "control"), exist_ok=True)
    srcs = {}
    for dst in ("pulse/n.json", "status/n.json", "status/n/replies/482", "status/n/events/e.json"):
        p = os.path.join(root, dst.replace("/", "_")); open(p, "w").write("{}"); srcs[dst] = p
        C.enqueue(p, dst)
    sent = []
    monkeypatch.setattr(C, "remote_put", lambda src, dst: (sent.append(dst), dst != "pulse/n.json")[1])
    shipped, dropped, failed = C.ship_outbox(budget_s=60)
    assert failed == 1 and shipped == 3 and {"status/n.json", "status/n/replies/482", "status/n/events/e.json"} <= set(sent)
    left = [json.load(open(os.path.join(C._outbox(), n)))["dst"] for n in os.listdir(C._outbox()) if n.endswith(".json")]
    assert left == ["pulse/n.json"]   # kept for the next pass, not dropped


def test_pulse_fields(root, monkeypatch):
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "CDIR", os.path.join(root, "control"))
    os.makedirs(os.path.join(root, "control"), exist_ok=True)
    st = os.path.join(root, "st.json"); open(st, "w").write('{"utc": "x", "tasks": {}}')
    C.pulse(st, upload=False)
    p = json.load(open(os.path.join(root, "control", "pulse.json")))
    assert p["source_policy"] == "private_first" and p["public_mirror_utc"] is None


def test_setenv_public_uploads():
    R.parse({"cmd": "setenv PUBLIC_UPLOADS=1", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 75)


def test_cli_still_dispatches_put(root):   # the CLI dict must still be called (a put reaches put(): here it fails without a token)
    r = subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "put", "/nonexistent", "x"], capture_output=True, text=True,
                       env={k: v for k, v in dict(os.environ, OPD_ROOT=root).items() if k != "HF_TOKEN"})
    assert r.returncode == 1 and "put failed" in r.stderr


def test_weights_deferred_while_paused(root):   # [ops-lead's A1 condition] no private call, one pending line, exit 0
    w = os.path.join(root, "w"); os.makedirs(w); open(os.path.join(w, "config.json"), "w").write("{}")
    with open(os.path.join(w, "model.safetensors"), "wb") as f:
        f.truncate(101 * 10 ** 6)                                         # sparse: 101 MB
    ck = tempfile.mkdtemp()
    env = {k: v for k, v in dict(os.environ, OPD_ROOT=root, CKPT_ROOT=ck).items() if k != "HF_TOKEN"}   # an upload attempt would fail: no token
    run = lambda src, dst, e=env: subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "put", src, dst], capture_output=True, text=True, env=e)
    r = run(w, "runs/E1/global_step_50/actor/huggingface")
    assert r.returncode == 0, r.stderr
    lines = open(os.path.join(root, "control", "weights_pending.txt")).read().splitlines()
    held = os.path.join(ck, "weights_pending", "runs/E1/global_step_50/actor/huggingface")
    assert len(lines) == 1 and lines[0].split("\t")[1:3] == ["runs/E1/global_step_50/actor/huggingface", held] and int(lines[0].split("\t")[3]) >= 101 * 10 ** 6
    import shutil
    shutil.rmtree(w + "_gone", ignore_errors=True); os.rename(w, w + "_gone"); shutil.rmtree(w + "_gone")   # upload_sync.sh deletes its copy ...
    assert os.path.getsize(os.path.join(held, "model.safetensors")) == 101 * 10 ** 6 and os.path.isfile(os.path.join(held, "config.json"))   # ... the held one stays
    os.makedirs(w); open(os.path.join(w, "config.json"), "w").write("{}")
    with open(os.path.join(w, "model.safetensors"), "wb") as f:
        f.truncate(101 * 10 ** 6)
    env2 = {k: v for k, v in env.items() if k != "CKPT_ROOT"}
    assert run(w, "runs/E2/global_step_50/actor/huggingface", env2).returncode == 1   # nowhere to hold it: the put fails, the caller keeps its files
    assert run(os.path.join(root, "hf_repos.txt"), "ckpts/x/a.json").returncode == 0 and len(open(os.path.join(root, "control", "weights_pending.txt")).read().splitlines()) == 2
    assert run(os.path.join(root, "hf_repos.txt"), "runs/E1/metrics.json").returncode == 1   # a small result still goes up (to the private repository)
    open(os.path.join(root, "control.env"), "w").write("export PUBLIC_UPLOADS=1\n")
    assert run(w, "runs/E1/global_step_50/actor/huggingface").returncode == 1                # not paused: uploaded as before
    assert len(open(os.path.join(root, "control", "weights_pending.txt")).read().splitlines()) == 2


def test_outbox_ships_the_control_plane_first(root, monkeypatch):   # [v2.5.9z51] status / pulse / receipts before a backlog
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "CDIR", os.path.join(root, "control"))
    os.makedirs(os.path.join(root, "control"), exist_ok=True)
    order = ["runs/E1/a.csv", "runs/E2/b.csv", "status/n/events/e.json", "runs/E3/c.csv", "pulse/n.json", "status/n.json", "status/n/replies/482"]
    for dst in order:
        p = os.path.join(root, dst.replace("/", "_")); open(p, "w").write("{}"); C.enqueue(p, dst)
    sent = []
    monkeypatch.setattr(C, "remote_put", lambda src, dst: (sent.append(dst), True)[1])
    C.ship_outbox(budget_s=60)
    assert sent == ["status/n/events/e.json", "pulse/n.json", "status/n.json", "status/n/replies/482", "runs/E1/a.csv", "runs/E2/b.csv", "runs/E3/c.csv"]
