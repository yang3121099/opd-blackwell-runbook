"""Result-repository backend for the remote node: Hugging Face (HF_TOKEN, HF_REPO) or ModelScope (MS_TOKEN, MS_REPO),
chosen by UPLOAD_BACKEND. Credentials are read from the environment only and never printed or written to disk.
  python remote.py check                 -> exit 0 if the backend is usable (token valid, repo reachable)
  python remote.py put <local> <remote>  -> upload a file or a folder to <remote> in the repo
The ModelScope path is not tested yet: check upload.log for UP_OK after the first checkpoint."""
import os, sys

BACKEND = os.environ.get("UPLOAD_BACKEND", "hf")


def hf():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO"]


def ms():
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(os.environ["MS_TOKEN"])
    return api, os.environ["MS_REPO"]


def check():
    if BACKEND == "hf":
        api, repo = hf(); api.whoami(); api.model_info(repo)
    else:
        api, repo = ms(); api.get_model(repo)


def put(src, dst):
    if BACKEND == "hf":
        api, repo = hf()
        if os.path.isdir(src):
            api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, repo_type="model", commit_message=dst)
        else:
            api.upload_file(repo_id=repo, path_or_fileobj=src, path_in_repo=dst, repo_type="model", commit_message=dst)
    else:
        api, repo = ms()
        if os.path.isdir(src):
            api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, commit_message=dst)
        else:
            api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, commit_message=dst)


if __name__ == "__main__":
    try:
        check() if sys.argv[1] == "check" else put(sys.argv[2], sys.argv[3])
    except Exception as e:   # never echo the exception text verbatim: it can contain request headers
        print(f"{BACKEND} {sys.argv[1]} failed: {type(e).__name__}", file=sys.stderr); sys.exit(1)
