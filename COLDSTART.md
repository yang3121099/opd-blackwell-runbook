# Cold start: an empty machine back to running (runbook v2.5.9z100; requirements v26 §11)

For a NEW machine whose disks hold nothing of ours (REBUILD.md covers a new container on the old share). Everything comes from
git (this runbook) and the two Hugging Face repositories. Target: the first GPU task within 4 h of pasting the command.

## Before
- The token in a file only you can read: `install -m 600 /dev/null <file>`, then put the token in it with an editor (never on a
  command line, never in the environment of your shell).
- The largest LOCAL disk for the runbook clone, --opd-root and the checkpoints (>= 1 TB free): the control plane (the runbook,
  control/, logs/) must sit on one local disk, never a network filesystem - the health check refuses it otherwise.

## The command (one block, under nohup: it runs for hours)
```bash
git clone https://github.com/<owner>/opd-blackwell-runbook <disk>/opd-blackwell-runbook && cd <disk>/opd-blackwell-runbook && setsid nohup bash scripts/cold_start.sh --token-file <0600 file> --opd-root <disk>/opd-run --ckpt-root <disk>/opd-ckpt --repos public=<owner/name>,private=<owner/name> --node <old node id, to keep its history> > <disk>/coldstart.out 2>&1 & sleep 5; tail -f <disk>/opd-run/logs/coldstart.log
```
`--node`: the old node's id keeps ctl.py, the status path and the control history; without it the machine gets a new id from its
host name (then ctl.py is pointed at the new id).

## What happens (STEP lines in logs/coldstart.log)
0. the token file (0600) read into the process only; paths.env (CKPT_ROOT, OPD_NODE), hf_repos.txt.
1. **control bootstrap before any loop**: control/applied_seq = the largest seq of control/<node>.yaml, so no old command runs;
   control/control.env.proposed = each setenv key's last value, printed, **not applied** - ops-lead reviews it, then the lines kept
   go to $OPD_ROOT/control.env by hand.
2. overnight.sh (OVERNIGHT_NO_LOOPS=1): base models, the cu128 venv, data, the vLLM tier probe, verify.sh.
3. the node's last status from the private repository: which tasks are done.
4. in parallel: the other models the unfinished tasks need (the manifest, regenerated from the pulled queue), venv-fa, venv-vl,
   venv-ifeval.
5. the probe labels from the private repository, checked against the Hub's hash; missing ones: rerun the probe, registered as a NEW
   label version.
6. which weights the unfinished tasks need are on the Hub (control/coldstart_weights.json; missing ones: those tasks wait).
7. health checks, all PASS or nothing starts (logs/coldstart_health.txt): GPUs, CKPT_ROOT >= 1 TB, both repositories answer, the
   token writes to the private repository (a probe under status/<node>/), the control plane on one local disk, a reused node id's
   last status older than 2 h (two machines never share an id), GitHub, the four venvs, verify.sh, applied_seq == the history's
   largest with no command run during the bootstrap, the outbox empty, no loop yet.
8. heartbeat and run_tasks, and one event "coldstart" with the skipped seqs.

A rerun skips what is done. COLDSTART_DRYRUN=1 installs, downloads and starts nothing (scripts/test_cold_start.sh).
After step 8 the executor checks from the project side: three fresh status rounds, one `ps` with its receipt.
