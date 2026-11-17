"""[v3.1] gpu_monitor.py: this node's own record of GPU utilisation against the machine's rule (Fable 11-17; Runming: "utilisation at
least 85 %, about 90 % normally, else the job is ended after two hours" - not enforced yet, so we measure ourselves first).

Every SAMPLE_S (10) s one `nvidia-smi` read of every GPU. Per GPU: the mean of the last minute, 10 minutes, 30 minutes and 2 hours,
the share of the last 2 hours spent at or above the floor, and who is on the card (a task of ours, the owner's filler, nothing).
  control/gpu_monitor/state.json   every 30 s: the table above, `low2h` = the GPUs whose 2-hour mean is under GPU_UTIL_FLOOR (85) -
                                   the rule as we understand it - and `warn30m` = those whose 30-minute mean is under it (the early
                                   warning: 90 minutes before the rule would act). The heartbeat puts it into the status as gpu_util.
  logs/gpu_util.csv                one line a minute: utc, then each GPU's 1-minute mean (local; kept for the record, 30 days)
  events                           gpu_util_warn / gpu_util_low when a GPU enters those lists, gpu_util_ok when both are empty again
A window counts only once it is at least 80 % full, so a fresh start raises nothing. Numbers only: no path, no command line.
Usage: python3 gpu_monitor.py [run] | once | status      (stdlib only; scripts/gpu_monitor.sh keeps one copy running)"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run"))
RUNBOOK = os.environ.get("RUNBOOK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MDIR = os.path.join(ROOT, "control", "gpu_monitor")
WINDOWS = (("u1m", 60), ("u10m", 600), ("u30m", 1800), ("u2h", 7200))


def floor():
    v = None
    try:
        for line in open(os.path.join(ROOT, "control.env")):
            m = re.fullmatch(r"export GPU_UTIL_FLOOR=([0-9]{2})", line.strip())
            if m:
                v = m.group(1)
    except OSError:
        pass
    v = v or os.environ.get("GPU_UTIL_FLOOR") or "85"
    return float(v) if v.isdigit() and 50 <= int(v) <= 99 else 85.0


def sample():
    """[(gpu, utilisation, memory used MiB)] or [] when nvidia-smi does not answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        w = [c.strip() for c in line.split(",")]
        if len(w) == 3 and w[0].isdigit() and w[1].isdigit():
            rows.append((int(w[0]), float(w[1]), float(w[2]) if w[2].isdigit() else 0.0))
    return rows


def window(q, now, span, fl=None):
    """(mean, full?, share at or above fl) of the samples of the last `span` seconds in q = [(t, util)]."""
    xs = [u for t, u in q if t >= now - span]
    if not xs:
        return None, False, None
    full = q[0][0] <= now - span * 0.8
    return round(sum(xs) / len(xs), 1), full, (round(sum(1 for u in xs if u >= fl) / len(xs), 3) if fl is not None else None)


def owners():
    """{gpu: "task" | "filler"}: a task.sh / ad-hoc evaluation of ours holds the card, or the owner's filler runs on it."""
    out = {}
    try:
        ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        ps = ""
    for m in re.finditer(r"scripts/(?:task|run_adhoc_eval)\.sh (\d+(?:,\d+)*) ", ps + " "):
        for g in m.group(1).split(","):
            out[int(g)] = "task"
    ds = [os.path.join(ROOT, "control", "gpu_fill")]
    try:   # [v3.6] gpu_fill.sh keeps its sessions in memory since v3.2: /dev/shm/opd_gpu_fill_<uid>_<cksum of OPD_ROOT>
        ck = subprocess.run(["cksum"], input=(ROOT + "\n").encode(), capture_output=True, timeout=10).stdout.split()[0].decode()
        ds.insert(0, f"/dev/shm/opd_gpu_fill_{os.getuid()}_{ck}")
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    d = next((x for x in ds if os.path.isdir(x)), ds[-1])
    try:
        for f in os.listdir(d):
            if f.endswith(".pid") and f[:-4].isdigit() and int(f[:-4]) not in out:
                try:
                    os.killpg(int(open(os.path.join(d, f)).read().strip()), 0)
                    out[int(f[:-4])] = "filler"
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    return out


def fill_name():
    """GPU_FILL_NAME of control.env (the owner's filler is known by this script file name), or None."""
    try:
        m = re.findall(r"^export GPU_FILL_NAME=([A-Za-z0-9_.-]+)$", open(os.path.join(ROOT, "control.env")).read(), re.M)
        return m[-1] if m else None
    except OSError:
        return None


def apps():
    """[v3.6] {gpu: [what runs on it]}: for every compute process nvidia-smi lists, the file names of its program and of its
    script ("python train.py"; file names only, no directory, no other argument), or "pid not visible" when the process is outside
    this container. 11-17: a borrowed machine came with one filler process on all eight cards and the status could not say so."""
    out = {}
    try:
        idx = {}
        for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.splitlines():
            i, u = [x.strip() for x in line.split(",")[:2]]
            idx[u] = int(i)
        for line in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 2 or not f[0].isdigit() or f[1] not in idx:
                continue
            try:
                argv = [a for a in open(f"/proc/{f[0]}/cmdline", "rb").read().decode("utf-8", "replace").split("\0") if a]
                script = next((a for a in argv[1:] if a.endswith(".py") or a.endswith(".sh")), None)
                name = " ".join(os.path.basename(a) for a in ([argv[0]] if argv else []) + ([script] if script else []))[:60] or "?"
            except OSError:
                name = "pid not visible"
            if name not in out.setdefault(idx[f[1]], []):
                out[idx[f[1]]].append(name)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return out


def fill_state():
    """[v3.6] what gpu_fill.sh knows: is filling on, is the filler's command known, did a restart of it work here."""
    def first(path, n):
        try:
            return " ".join(open(path).read().split()[:n]) or None
        except OSError:
            return None
    try:
        on = re.findall(r"^export GPU_FILL=([01])$", open(os.path.join(ROOT, "control.env")).read(), re.M)
    except OSError:
        on = []
    c = os.path.join(ROOT, "control")
    t = first(os.path.join(c, "gpu_fill_selftest"), 4)
    return {"on": bool(on) and on[-1] == "1", "name": fill_name(),
            "command": "learned" if os.path.exists(os.path.join(c, "gpu_fill_secret.seen")) else "stored" if os.path.exists(os.path.join(c, "gpu_fill_secret.cmd")) else None,
            "selftest": (t.split()[0] + " " + " ".join(t.split()[2:])) if t else None}


def summarize(qs, now, fl):
    own, ap, fn = owners(), apps(), fill_name()
    st = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "floor": fl, "gpus": {}, "low2h": [], "warn30m": [], "fill": fill_state()}
    for g in sorted(qs):
        names = ap.get(g, [])
        row = {"on": own.get(g) or ("filler" if fn and any(n.split()[-1] == fn for n in names) else "other" if names else "none")}
        if names:
            row["apps"] = names[:3]
        for name, span in WINDOWS:
            mean, full, share = window(qs[g], now, span, fl if name == "u2h" else None)
            row[name] = mean
            if name == "u2h":
                row["share2h_at_floor"] = share
                if full and mean is not None and mean < fl:
                    st["low2h"].append(g)
            if name == "u30m" and full and mean is not None and mean < fl:
                st["warn30m"].append(g)
        st["gpus"][str(g)] = row
    means = [r["u10m"] for r in st["gpus"].values() if r["u10m"] is not None]
    st["node_u10m"] = round(sum(means) / len(means), 1) if means else None
    return st


def event(kind, detail):
    try:
        subprocess.Popen([sys.executable, os.path.join(RUNBOOK, "scripts", "control.py"), "event", "node", kind, detail],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass


def write_state(st):
    os.makedirs(MDIR, exist_ok=True)
    tmp = os.path.join(MDIR, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, os.path.join(MDIR, "state.json"))


def run(once=False):
    os.makedirs(MDIR, exist_ok=True); os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    step = float(os.environ.get("SAMPLE_S", "10"))
    qs, t_state, t_csv, last = {}, 0.0, 0.0, (frozenset(), frozenset())
    csv = os.path.join(ROOT, "logs", "gpu_util.csv")
    while True:
        now = time.time()
        for g, u, _ in sample():
            q = qs.setdefault(g, [])
            q.append((now, u))
            while q and q[0][0] < now - 7200:
                q.pop(0)
        if once or now - t_state >= 30:
            t_state = now
            st = summarize(qs, now, floor())
            write_state(st)
            cur = (frozenset(st["low2h"]), frozenset(st["warn30m"]))
            if cur != last:
                if cur[0] - last[0]:
                    event("gpu_util_low", "2-hour mean under %g on GPU %s" % (st["floor"], ",".join("%d (%s)" % (g, st["gpus"][str(g)]["u2h"]) for g in sorted(cur[0]))))
                elif cur[1] - last[1]:
                    event("gpu_util_warn", "30-minute mean under %g on GPU %s" % (st["floor"], ",".join("%d (%s)" % (g, st["gpus"][str(g)]["u30m"]) for g in sorted(cur[1]))))
                elif not cur[0] and not cur[1]:
                    event("gpu_util_ok", "every GPU's 30-minute and 2-hour means are at or over the floor")
                last = cur
            if once:
                print(json.dumps(st)); return 0
        if now - t_csv >= 60 and qs:
            t_csv = now
            try:
                if os.path.exists(csv) and os.path.getsize(csv) > 30 * 1440 * 120:   # about 30 days of lines: start again
                    os.replace(csv, csv + ".1")
                with open(csv, "a") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)) + "," + ",".join(str(window(qs[g], now, 60)[0]) for g in sorted(qs)) + "\n")
            except OSError:
                pass
        time.sleep(step)


def main(argv):
    if argv[:1] == ["status"]:
        try:
            print(open(os.path.join(MDIR, "state.json")).read())
        except OSError:
            print("{}")
        return 0
    if argv[:1] == ["once"]:
        return run(once=True)
    if argv in ([], ["run"]):
        return run()
    print(__doc__.split("Usage:")[1].strip()); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
