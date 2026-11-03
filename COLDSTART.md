# Cold start: a new machine, connected first, everything else remote (runbook v2.5.9z127)

For a NEW machine (nothing of ours on its disks). The person at the machine pastes one line and types two tokens; from the moment
the line prints `node <id> online` the project installs, checks, repairs and runs this machine through control commands, and that
person is never asked again (next_machine_selfserve_20261102). Targets: status, pulse and the GitHub heartbeat within 5 min; the
weights downloading within 10 min; the first arm within 2.5 h; every GPU busy within 6 h.

## The line (the only thing pasted)
```bash
R=https://github.com/yang3121099/opd-blackwell-runbook.git; d=$(mktemp -d); { git clone -q $R $d 2>/dev/null || { rm -rf $d; d=$(mktemp -d); p=$(grep -hoE "^(export +)?(https_proxy|HTTPS_PROXY)=['\"]?[^'\" ]+" /etc/environment /etc/profile /etc/profile.d/*.sh ~/.bashrc ~/.profile 2>/dev/null | head -1 | sed -E "s/^(export +)?[A-Za-z_]+=//; s/['\"]//g"); [ -n "$p" ] && https_proxy=$p git clone -q $R $d; } || { echo "STOP: git cannot reach github.com from this machine: ask how git reaches the internet here"; false; }; } && bash $d/scripts/bootstrap.sh --profile 80g
```
(The clone into a temporary directory only fetches bootstrap.sh. When git cannot reach GitHub directly, the line retries once
through the first https_proxy in /etc/environment, /etc/profile(.d), ~/.bashrc or ~/.profile, value not printed; else it prints
STOP. The owner's copy with the expected output: the lab's handoff/newnode_paste_20261102.md.)
- It asks for the Hugging Face token and then the GitHub heartbeat token, each typed and not shown (`read -rs`; nothing on a
  command line, in the shell history or in a file anyone else can read). Enter at the second one skips the GitHub heartbeat.
- `--base <dir>`: the disk to use (default: the largest writable local disk). `--profile <name>[,<name>]`: the node class
  (profiles/*.env: `80g` for 80 GB cards). `--role rehearsal`: a rehearsal's commit limits (profiles/rehearsal.env).
- Rerunning the line is safe: every step looks at what is there first.

## What bootstrap.sh does (2-5 min, then `node <id> online; everything else is remote`)
B2 the base directory; B0 the HF token (0600); B0b the GitHub heartbeat token, stored for git alone (0600); B1 direct or a working
proxy found on the machine (values never printed); B1b one `git ls-remote` of the heartbeat repository; B3 the runbook clone and
the node id (a hash of the host name); B4 the control venv (huggingface_hub, pyyaml; no torch); B5 cold_start.sh step 1 (no old
command can run), node.env, the profile's keys into control.env (a key already there is kept), heartbeat every 120 s, **dispatch
frozen**; B6 heartbeat + run_tasks (+ watchdog, uploader, gh_heartbeat), the first status, the first GitHub push.

## From there, remote only (the executor pushes these; they are pre-approved for T0)
| Command | What it does |
|---|---|
| `env-probe` | GPUs, driver, disks, network, an old base or not: into status |
| `setenv K=V` | the differences from the profile |
| `mounts` | read-only mount probe (shared disks) |
| `setup all` | cold_start.sh steps 2-7 in the background, each step in status.setup_state; a STOP shows the step log's last lines |
| `run FETCH-<name>` / `run FETCH-all` | the queue's `#fetch` lines pulled while dispatch is frozen and before the main venv (control venv, 32 connections for a start, no GPU, never task.sh, so `setup` is never refused because of them) |
| `run PROBE-TAIL` | the canary, once the main venv is ready |
| (z134) a local copy first | every `#fetch` pull and setup's model pulls look for the same files on this machine's disks first: an old base the bootstrap line recorded (node.env OLD_BASE_CANDS / FROM_OPD_ROOT, or FETCH_LOCAL_ROOTS in node.env, never setenv), its opd-ckpt and DR mirror; a source with every listed file at the listed size is copied (read only), the copy checked like a download (sha256 of LFS files), the rest from the Hub; FETCH_LOCAL=0 turns it off |
| `dispatch on` | accepted only when the **main venv** imports torch / vllm / verl (control/venv_ready) **and the canary PROBE-TAIL passed on this node**; nothing else |
| `tail <log>`, `setup <N>`, `restart-loops` | repair: read the failing step's log, `setenv` the fix, rerun that step |

## verify reports, it does not stop (z127)
verify.sh's five checks (G2 unit tests; GPU / vLLM; the 3-step smoke; the upload path; the control self-test) go into
status.verify (control/verify_state.json, "ok" = the first four passed). On a connect-first node a FAIL stops neither setup step 2
nor the step 7 health checks nor `dispatch on`. It holds only the queue rows that declare `#needs verify <task>`; they start once
verify passes (`verify` reruns it). While VENV=fa is set and venv-fa is not built yet, the smoke runs on the main venv.

## GitHub blocked, Hugging Face open (z135)
The usual case for machines in mainland China. The runbook has a git mirror on Hugging Face
(https://huggingface.co/yang31210999/opd-blackwell-runbook, public, the same main: the Mac pushes it after every push to GitHub).
The paste page's third line clones the mirror and runs bootstrap with `RB_URL=<the mirror>`: B1 then asks the mirror's info/refs
and the HF API instead of GitHub, B1b skips its check when GitHub does not answer, and node.env carries RB_URL and
OPD_CONTROL_ORIGIN, so run_tasks' git pull and control.py's origin check go through the mirror. B1 always prints the HTTP codes
of GitHub, the runbook's host, the HF API and PyPI; without PyPI, setup step 2 can only link an old base's venv
(FROM_OPD_ROOT + LINK_VENV=1). Still on GitHub (a second stage, only without an old base): G-OPD's clone, the flash-attn wheel,
the aria2 binary.

## The one case that needs the person again
bootstrap.sh prints "ask the machine's owner how git / pip reach the internet here": neither direct nor any proxy found on the
machine reaches GitHub and Hugging Face.

## Without bootstrap (older flow, kept for a machine that cannot run it)
`setsid nohup bash scripts/cold_start.sh --token-file <0600 file> --opd-root <disk>/opd-run --ckpt-root <disk>/opd-ckpt --repos
public=<owner/name>,private=<owner/name>` runs steps 0-8 in one go: there a verify FAIL or any health check FAIL stops before the
loops start. COLDSTART_DRYRUN=1 installs, downloads and starts nothing (scripts/test_cold_start.sh).
