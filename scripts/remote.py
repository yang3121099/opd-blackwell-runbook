"""Result-repository backend for the remote node: Hugging Face (HF_TOKEN, HF_REPO) or ModelScope (MS_TOKEN, MS_REPO),
chosen by UPLOAD_BACKEND. Credentials are read from the environment only and never printed or written to disk.
  python remote.py check                 -> exit 0 if the backend is usable (token valid, repo reachable)
  python remote.py put <local> <remote>  -> upload a file or a folder to <remote> in the repo
  python remote.py done <task>           -> exit 0 if runs/<task>/.done exists in the repo (HF only; ModelScope: no)
The ModelScope path is not tested yet: check upload.log for UP_OK after the first checkpoint."""
import os, sys

BACKEND = os.environ.get("UPLOAD_BACKEND", "hf")


def hf():
    from huggingface_hub import HfApi
    # the result repository is always reached at huggingface.co (v2.2): HF_ENDPOINT (e.g. the read-only hf-mirror.com)
    # only speeds up downloads of public models and data and must not redirect uploads; HF_UPLOAD_ENDPOINT overrides
    return (HfApi(endpoint=os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co"), token=os.environ["HF_TOKEN"]),
            os.environ["HF_REPO"])


def ms():
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(os.environ["MS_TOKEN"])
    return api, os.environ["MS_REPO"]


def check():
    if BACKEND == "hf":
        api, repo = hf(); api.whoami()
        # create the result repository if it does not exist yet (public by default; REQUIRE_PRIVATE=1 creates it private
        # and refuses a public one). A token limited to one repository may not be allowed to create it: then the project
        # lead creates it on the website first.
        need_private = os.environ.get("REQUIRE_PRIVATE", "0") == "1"
        api.create_repo(repo, repo_type="model", private=need_private, exist_ok=True)
        if need_private:
            assert api.model_info(repo).private, "REQUIRE_PRIVATE=1 but the result repository is public"
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


def done(task):
    if BACKEND != "hf":
        sys.exit(1)
    api, repo = hf()
    sys.exit(0 if api.file_exists(repo, f"runs/{task}/.done", repo_type="model") else 1)


if __name__ == "__main__":
    try:
        {"check": lambda: check(), "done": lambda: done(sys.argv[2])}.get(sys.argv[1], lambda: put(sys.argv[2], sys.argv[3]))()
    except Exception as e:   # never echo the exception text verbatim: it can contain request headers
        print(f"{BACKEND} {sys.argv[1]} failed: {type(e).__name__}", file=sys.stderr); sys.exit(1)
