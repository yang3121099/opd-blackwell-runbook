"""[v2.5.9z100] cold_start.sh's local checks (ops-lead's A1 10-27): the control plane on ONE local disk, and an old node id reused
only when that node is silent.

  localdisk <path> [<path> ...]     every path on the same device, none on a network filesystem (/proc/mounts; COLDSTART_MOUNTS
                                    for tests): exit 0 / 1, one line why
  silent <status.utc file> <hours>  the reused node's last status is older than <hours> (no status: silent): exit 0 / 1
"""
import calendar
import os
import sys
import time

NETFS = ("nfs", "nfs4", "cifs", "smb", "smbfs", "smb3", "ceph", "lustre", "gpfs", "beegfs", "glusterfs", "9p", "afs", "davfs")


def mounts():
    try:
        rows = [l.split() for l in open(os.environ.get("COLDSTART_MOUNTS", "/proc/mounts"))]
    except OSError:
        return []
    return sorted(((r[1], r[2]) for r in rows if len(r) >= 3), key=lambda x: -len(x[0]))


def fstype(path, table):
    p = os.path.realpath(path)
    for mp, t in table:
        if p == mp or p.startswith(mp.rstrip("/") + "/"):
            return t
    return "unknown"


def localdisk(paths):
    real = [os.path.realpath(p) for p in paths]
    devs = {os.stat(p).st_dev for p in real}
    table = mounts()
    net = [(p, fstype(p, table)) for p in real if fstype(p, table).split(".")[0] in NETFS or fstype(p, table).startswith("fuse")]
    if len(devs) > 1:
        print("LOCALDISK FAIL: the control plane spans more than one device"); return 1
    if net:
        print(f"LOCALDISK FAIL: on a network filesystem ({net[0][1]})"); return 1
    print("LOCALDISK PASS"); return 0


def silent(utc_file, hours):
    try:
        utc = open(utc_file).read().strip()
    except OSError:
        utc = ""
    if not utc:
        print("SILENT PASS: no earlier status"); return 0
    age = (time.time() - calendar.timegm(time.strptime(utc[:19], "%Y-%m-%dT%H:%M:%S"))) / 3600
    if age < float(hours):
        print(f"SILENT FAIL: the node's last status is {age:.1f} h old (< {hours} h): it may still be alive"); return 1
    print(f"SILENT PASS: last status {age:.1f} h old"); return 0


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    sys.exit(localdisk(a) if c == "localdisk" else silent(*a))
