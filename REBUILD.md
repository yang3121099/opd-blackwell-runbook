# Rebuilding a node after its container is lost (runbook v2.5.7a)

What survives and what does not. `$OPD_ROOT` and the runbook clone live on a network share, so a new container with the
same mounts finds them (runs/ logs and configs, data/, models/, venv-fa, control state, hf_repos.txt, paths.env,
models.env). The container disk is lost: `$HOME` (uv's python, the HF cache), CKPT_ROOT (`$HOME/opd-ckpt`: resume
states, local bf16 copies, fetched models), the running processes and the environment (HF_TOKEN).

Since v2.5.7 / v2.5.7a nothing a task needs is only on the container disk:
- resume states: every training mirrors its newest one every 25 steps to `<share base>/opd-dr/dr/<task>/` (v2.5.7);
- the venv in use, uv's python and G-OPD: `<share base>/opd-env/` (`persist-env`, v2.5.7a), `$OPD_ROOT/venv` links there;
- the three models: `$OPD_ROOT/models/Qwen3-{0.6B,1.7B,8B}` (`persist-env`), named in models.env;
- E12b starts: E10b finals from the public result repository, SFT-P1's final from `opd-dr/keep/` (`dr-keep`);
- the small control files also in `opd-dr/state/` (every heartbeat).

## Steps (the person at the node; target <= 10 min of their time)
1. New container with the same mounts (the share with `<share base>`), GPUs visible (`nvidia-smi`).
2. In a shell:
   ```bash
   export OPD_ROOT=<share base>/opd-run
   export HF_TOKEN=<the result repositories' token>        # never written to a file by the runbook
   cd <share base>/opd-blackwell-runbook && git pull       # or: git clone https://github.com/yang3121099/opd-blackwell-runbook
   ```
3. Check the environment (seconds): `$OPD_ROOT/venv/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"`
   prints the versions (the venv and its python are on the share). If it fails, run `bash scripts/persist_env.sh` on a
   machine where the old container still runs, or rebuild the venv with `bash scripts/install_uv.sh` (~15 min).
4. Start everything: `setsid nohup bash scripts/overnight.sh > /dev/null 2>&1 &`. It finds the models in
   `$OPD_ROOT/models`, reuses the venv and data, recreates CKPT_ROOT, probes vLLM, runs verify.sh, then starts the
   heartbeat and run_tasks. Unfinished trainings restart and copy their mirrored resume state back (`DR_RESTORED` in
   their train.log); evaluations re-fetch their models from the result repository.
5. Check from the project side: `python ctl.py <node> status` shows the runbook version, `alerts []`, the tasks running
   again, and `dr` with the mirrored steps.
Loss: at most 25 steps per training (plus the time of verify.sh).

## Drill
`rebuild-drill` (control verb) runs steps 2-4's checks without a GPU in a temporary directory on the share (clone, a
fresh `$OPD_ROOT` from the share copies and `opd-dr/state`, python imports, the three models complete, a mirrored
resume state copied back, env.sh / dr.sh of the clone) and deletes it again; `logs/rebuild_drill.json` has the seconds
per step. Nothing in use is touched and nothing is started.
