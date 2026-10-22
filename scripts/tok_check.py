"""[v2.5.9z24] tok_check.py (TOKCHECK-X5; ops-lead 10-22, Q-057): zero-GPU facts of freshly pulled models, from their files only.

    python tok_check.py <OPD_ROOT> <report.json> <label>=<model dir name> ...

Per model (models/<name>): the repo, size and time of its .fetch_ok; config.json (model_type, vocab_size = embedding rows, the
bos / eos / pad ids), generation_config.json's ids, tokenizer_config.json's special tokens (and their ids in the tokenizer),
tokenizer.json (base vocab, added tokens, distinct ids), whether the chat template mentions <think> (and the ids of <think> /
</think>); with transformers importable, also len(tokenizer), its special ids and the tail of a chat prompt with the
generation prompt added. The pull: the last successful fetch_model.py line of logs/fetch_<name>.log (bytes, seconds, MB/s;
start = the .fetch_ok time minus its seconds) and, over all models, the combined rate (bytes / the span from the first start to
the last end) and the seconds all of them ran at once. Pairs (the first label against each other one): id for id over the
union of ids (tokenizer.json vocab and added tokens): same, different, only in one; the first differences; the base vocab alone.
Names only, never a path."""
import calendar
import json
import os
import sys
import time


def load(p, default=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def tok_str(v):   # a special token in tokenizer_config.json: a string or {"content": ...}
    return v.get("content") if isinstance(v, dict) else v


def id_maps(tj):
    """id -> token over the model vocab and the added tokens; the base vocab alone; the added tokens alone."""
    vocab = ((tj or {}).get("model") or {}).get("vocab")
    base = {int(i): t for t, i in vocab.items()} if isinstance(vocab, dict) else {}
    added = {int(a["id"]): a["content"] for a in (tj or {}).get("added_tokens") or [] if "id" in a and "content" in a}
    return {**base, **added}, base, added


def epoch(utc):
    try:
        return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def last_ok_line(path):
    r = None
    try:
        for x in open(path, errors="ignore"):
            x = x.strip()
            if x.startswith("{"):
                try:
                    j = json.loads(x)
                except ValueError:
                    continue
                if j.get("ok") and not j.get("cached"):
                    r = j
    except OSError:
        pass
    return r


def template_text(d, tc):
    t = tc.get("chat_template") or ""
    if isinstance(t, list):
        t = " ".join(x.get("template", "") for x in t if isinstance(x, dict))
    try:
        t += open(os.path.join(d, "chat_template.jinja")).read()
    except OSError:
        pass
    return t


def model_facts(root, name):
    d = os.path.join(root, "models", name)
    ok = load(os.path.join(d, ".fetch_ok"), {}) or {}
    cfg = load(os.path.join(d, "config.json"), {}) or {}
    gen = load(os.path.join(d, "generation_config.json"), {}) or {}
    tc = load(os.path.join(d, "tokenizer_config.json"), {}) or {}
    ids, base, added = id_maps(load(os.path.join(d, "tokenizer.json")))
    rev = {t: i for i, t in ids.items()}
    tmpl = template_text(d, tc)
    files = ok.get("files") or []
    special = {k: tok_str(tc.get(k)) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
    f = {"repo": ok.get("repo"), "bytes": sum(int(x[1]) for x in files if isinstance(x, (list, tuple)) and len(x) > 1 and str(x[1]).isdigit()),
         "fetched_utc": ok.get("utc"),
         "config": {k: cfg.get(k) for k in ("model_type", "architectures", "vocab_size", "hidden_size", "num_hidden_layers", "tie_word_embeddings",
                                            "torch_dtype", "max_position_embeddings", "rope_theta", "bos_token_id", "eos_token_id", "pad_token_id")},
         "generation_config": {k: gen.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")},
         "special_tokens": special, "special_token_ids": {k: rev.get(v) for k, v in special.items() if v is not None},
         "add_bos_token": tc.get("add_bos_token"),
         "tokenizer": {"base_vocab": len(base), "added_tokens": len(added), "distinct_ids": len(ids), "max_id_plus_1": max(ids) + 1 if ids else None},
         "chat_template": bool(tmpl), "template_has_think": "<think>" in tmpl,
         "think_ids": {"<think>": rev.get("<think>"), "</think>": rev.get("</think>")}}
    if isinstance(cfg.get("vocab_size"), int) and ids:
        f["embedding_rows_minus_tokenizer"] = cfg["vocab_size"] - (max(ids) + 1)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(d)
        f["hf"] = {"len": len(tok), "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id}
        try:
            p = tok.apply_chat_template([{"role": "user", "content": "What is 1+1?"}], tokenize=False, add_generation_prompt=True)
            f["hf"]["prompt_tail"] = p[-80:]
            f["hf"]["prompt_ends_in_think"] = p.rstrip().endswith("<think>")
        except Exception as e:   # no template, or one that needs more
            f["hf"]["prompt_tail"] = f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:
        f["hf"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    pull = last_ok_line(os.path.join(root, "logs", f"fetch_{name}.log"))
    end = epoch(ok.get("utc"))
    if pull and end is not None and pull.get("seconds") is not None:
        f["pull"] = {k: pull.get(k) for k in ("bytes", "seconds", "mb_s", "files")}
        f["pull"]["start"], f["pull"]["end"] = end - float(pull["seconds"]), end
    return f, ids, base


def compare(a, b, a_base, b_base):
    both = set(a) & set(b)
    diff = sorted(i for i in both if a[i] != b[i])
    return {"ids_compared": len(set(a) | set(b)), "same": len(both) - len(diff), "different": len(diff),
            "only_in_first": len(set(a) - set(b)), "only_in_second": len(set(b) - set(a)),
            "identical": not diff and set(a) == set(b), "base_vocab_identical": a_base == b_base,
            "different_id_range": [diff[0], diff[-1]] if diff else None,
            "first_differences": [[i, a[i], b[i]] for i in diff[:12]]}


def main(argv):
    root, out = argv[0], argv[1]
    pairs = [x.split("=", 1) for x in argv[2:]]
    r = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}, "pairs": {}}
    maps = {}
    for label, name in pairs:
        f, ids, base = model_facts(root, name)
        r["models"][label] = {"name": name, **f}
        maps[label] = (ids, base)
    first = pairs[0][0]
    for label, _ in pairs[1:]:
        r["pairs"][f"{first}_vs_{label}"] = compare(maps[first][0], maps[label][0], maps[first][1], maps[label][1])
    pulls = {k: m["pull"] for k, m in r["models"].items() if m.get("pull")}
    if pulls:
        s, e = min(p["start"] for p in pulls.values()), max(p["end"] for p in pulls.values())
        tot = sum(p["bytes"] or 0 for p in pulls.values())
        r["pull_combined"] = {"models": len(pulls), "bytes": tot, "span_s": round(e - s, 1), "mb_s": round(tot / 1e6 / (e - s), 2) if e > s else None,
                              "all_at_once_s": round(max(0.0, min(p["end"] for p in pulls.values()) - max(p["start"] for p in pulls.values())), 1)}
        for p in pulls.values():
            p["start"], p["end"] = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(x)) for x in (p["start"], p["end"])]
    json.dump(r, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: {"vocab_size": m["config"]["vocab_size"], "tokenizer": m["tokenizer"], "think": m["template_has_think"]} for k, m in r["models"].items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
