"""[v2.5.9z63] run_vlm.sh (E15-VLM-P5 and PROBE-VLM's smoke) loads its data in the main process unless VLM_DL_WORKERS says otherwise:
the 10-22 smoke's DataLoader workers died of a bus error writing to a nearly full /dev/shm."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def test_dataloader_workers_zero_by_default():
    src = open(os.path.join(HERE, "run_vlm.sh")).read()
    assert "data.dataloader_num_workers=${VLM_DL_WORKERS:-0}" in src
    i = src.index("data.dataloader_num_workers")
    assert src.index("data.train_files=") < i and src[src.rindex("\n", 0, i) - 1] == "\\"   # in the training command's data block
    assert subprocess.run(["bash", "-n", os.path.join(HERE, "run_vlm.sh")]).returncode == 0
