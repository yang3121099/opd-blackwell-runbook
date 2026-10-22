"""[v2.5.8] Tests of the control plane's outbox (control.py enqueue / queue_put / ship_outbox; design 4b.2 with the
reviewer's A1 additions): coalescing, time order, a failure ends the pass, demotion after 3 failures, the direct upload
without a live uploader, and receipts / events going through it. The stub remote.py of test_control copies into a
directory instead of uploading. Run: python -m pytest -q scripts/test_outbox.py"""
import json
import os
import time

from test_control import env  # noqa: F401  (shared fixture)


def alive(c):
    open(c._p("loop_uploader"), "w").write(str(int(time.time())))


def items(c):
    d = c._outbox()
    return [json.load(open(os.path.join(d, n))) for n in sorted(os.listdir(d)) if n.endswith(".json") and not n.startswith(".")] if os.path.isdir(d) else []


def test_queue_put_direct_without_uploader(env):
    c = env.c; f = env.root / "a.json"; f.write_text("{}")
    assert c.queue_put(str(f), "status/testnode/x.json") and (env.remote / "status/testnode/x.json").exists() and items(c) == []


def test_coalescing_and_order(env):
    c = env.c; alive(c)
    for n in ("s", "r1", "e1"):
        (env.root / n).write_text(n)
    c.queue_put(str(env.root / "s"), "status/testnode.json")
    c.queue_put(str(env.root / "r1"), "status/testnode/replies/1.json")
    c.queue_put(str(env.root / "s"), "status/testnode.json")              # newer copy: replaces the first, goes to the end
    c.queue_put(str(env.root / "e1"), "status/testnode/events/e1.json")
    assert [i["dst"] for i in items(c)] == ["status/testnode/replies/1.json", "status/testnode.json", "status/testnode/events/e1.json"]
    (env.root / "s").write_text("newest")                                  # the content is read when it ships
    assert c.ship_outbox() == (3, 0, 0) and items(c) == []
    assert (env.remote / "status/testnode.json").read_text() == "newest" and (env.remote / "status/testnode/events/e1.json").exists()


def test_failure_ends_the_pass_then_demotes(env, monkeypatch):   # [v2.5.9z50] the pass goes on past a failed item (was: ended there)
    c = env.c; alive(c)
    for n in ("a", "b", "r8"):
        (env.root / n).write_text(n)
    c.queue_put(str(env.root / "a"), "status/testnode/replies/7.json"); c.queue_put(str(env.root / "b"), "status/testnode/events/b.json")
    c.queue_put(str(env.root / "r8"), "status/testnode/replies/8.json")
    real = c.remote_put
    monkeypatch.setattr(c, "remote_put", lambda s, d: False if d.endswith("7.json") else real(s, d))
    assert c.ship_outbox() == (2, 0, 1) and [i["dst"] for i in items(c)] == ["status/testnode/replies/7.json"]
    assert items(c)[0]["fails"] == 1 and (env.remote / "status/testnode/events/b.json").exists() and (env.remote / "status/testnode/replies/8.json").exists()
    c.queue_put(str(env.root / "r8"), "status/testnode/replies/9.json")
    c.ship_outbox(); c.ship_outbox()                                       # the third failure moves it to the end, its count anew
    assert [(i["dst"], i["fails"]) for i in items(c)] == [("status/testnode/replies/7.json", 0)]


def test_missing_src_dropped(env):
    c = env.c; alive(c); (env.root / "gone").write_text("x")
    c.queue_put(str(env.root / "gone"), "status/testnode/events/g.json"); (env.root / "gone").unlink()
    assert c.ship_outbox() == (0, 1, 0) and items(c) == []


def test_receipt_and_event_go_through_the_outbox(env):
    c = env.c; alive(c)
    c.receipt(5, {"cmd": "ps", "reason": "t"}, "ok", "done"); c.event("node", "x_test", "d")
    dst = sorted(i["dst"] for i in items(c))
    assert dst[0].startswith("status/testnode/events/") and dst[1] == "status/testnode/replies/5.json"
    assert not (env.remote / "status/testnode/replies/5.json").exists()    # nothing uploaded until the uploader ships
    c.ship_outbox(); assert (env.remote / "status/testnode/replies/5.json").exists()


def test_outbox_off(env, monkeypatch):
    c = env.c; alive(c); monkeypatch.setenv("OUTBOX", "0"); (env.root / "o").write_text("o")
    assert c.queue_put(str(env.root / "o"), "status/testnode/o.json") and items(c) == [] and (env.remote / "status/testnode/o.json").exists()


def test_lru_clean(env, monkeypatch):   # [v2.5.8] off unless set; oldest eligible first; never a final, unuploaded or running step
    c = env.c; evs = []
    monkeypatch.setattr(c, "event", lambda t, k, d="": evs.append((t, k, d)))
    monkeypatch.setattr(c, "running_pids", lambda t=None: [1] if t == "E-run" else [])
    def step(t, n, fp=0, bf=0, up=True, age=0):
        sd = env.root / "runs" / t / "ckpt" / f"global_step_{n}"
        for sub, b in (("huggingface", fp), ("huggingface_bf16", bf)):
            if b:
                d = sd / "actor" / sub; d.mkdir(parents=True, exist_ok=True); (d / "model.safetensors").write_bytes(b"x" * b)
                os.utime(d, (time.time() - age, time.time() - age))
        sd.mkdir(parents=True, exist_ok=True)
        if up:
            (sd / ".uploaded").write_text("")
    step("E-a", 25, fp=100, bf=50, age=300); step("E-a", 50, fp=100, age=200); step("E-a", 100, fp=100, bf=50, age=100)   # 100 = final
    step("E-b", 10, fp=100, up=False, age=400); step("E-run", 5, fp=100, age=500)
    assert c.lru_clean() == 0.0 and evs == []                                       # unset: nothing
    total = 100 + 50 + 100 + 100 + 50 + 100 + 100                                   # 600 bytes
    freed = c.lru_clean(limit_gb=(total - 150) / 1e9)                               # 150 bytes too many
    gone = [e[2].split(" ")[0] for e in evs]
    assert gone == ["global_step_25/actor/huggingface", "global_step_50/actor/huggingface"] and freed == 0.0   # GB, rounded
    assert (env.root / "runs/E-a/ckpt/global_step_25/actor/huggingface_bf16/model.safetensors").exists()
    assert (env.root / "runs/E-a/ckpt/global_step_100/actor/huggingface/model.safetensors").exists()      # final kept
    assert (env.root / "runs/E-b/ckpt/global_step_10/actor/huggingface/model.safetensors").exists()       # not uploaded
    assert (env.root / "runs/E-run/ckpt/global_step_5/actor/huggingface/model.safetensors").exists()      # running
