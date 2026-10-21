"""The VLM teacher's inputs for OPD (runbook [v2.5.9z], Q-051; the reviewer's ruling 10-22: the student's image-expanded prompt ids +
the teacher template's prefix + the response, 3-D rope positions, the same multi-modal inputs).

G-OPD's prepare_ref_model_inputs re-tokenizes the raw prompt as text with the teacher's tokenizer: for Qwen3-VL that loses the image
(the placeholder is not expanded, the positions are 1-D, no pixel values reach the teacher). With OPD_VLM_REF=1 the hook
(vlmref_patch.py) returns build_batch() instead: the teacher reads EXACTLY the student's prompt ids (the image tokens as the student's
processor expanded them: PROBE-VLM (a2) checks the two processors and tokenizers are equal field by field), then the teacher template's
generation prefix (OPD_VLM_REF_SUFFIX, default '<think>\\n': Qwen3-VL-Thinking's template opens the think block, as P5's teacher
prompt), then the response. Positions: the student's own (bs, C, P) / (bs, P) prompt positions; the k suffix tokens continue every
channel from the last prompt position (+1 .. +k; text tokens advance the three rope channels together); the response positions are
the student's + k. multi_modal_inputs stay in the batch and the reference forward (dp_actor._forward_micro_batch) uses them with
ref_input_ids, as it does for the actor.
"""
import os


def build(input_ids, attention_mask, position_ids, responses, response_mask, suffix_ids, ops):
    """(ref_input_ids, ref_attention_mask, ref_position_ids). Tensors: input_ids / attention_mask (bs, P+R), position_ids (bs, P+R)
    or (bs, C, P+R), responses / response_mask (bs, R). ops: cat(list, dim), full(shape, value, like), arange(k, like)."""
    bs, total = input_ids.shape[0], input_ids.shape[1]
    r = responses.shape[1]
    p = total - r
    k = len(suffix_ids)
    suf = ops["full"]((bs, k), 0, input_ids)
    for j, t in enumerate(suffix_ids):
        suf[:, j] = t
    ids = ops["cat"]([input_ids[:, :p], suf, responses], 1)
    mask = ops["cat"]([attention_mask[:, :p], ops["full"]((bs, k), 1, attention_mask), response_mask], 1)
    step = ops["arange"](k, position_ids) + 1   # 1 .. k
    if position_ids.ndim == 3:
        last = position_ids[:, :, p - 1:p]                       # (bs, C, 1)
        mid = last + step.reshape(1, 1, k)
        pos = ops["cat"]([position_ids[:, :, :p], mid, position_ids[:, :, p:] + k], 2)
    else:
        last = position_ids[:, p - 1:p]
        mid = last + step.reshape(1, k)
        pos = ops["cat"]([position_ids[:, :p], mid, position_ids[:, p:] + k], 1)
    return ids, mask, pos


def torch_ops():
    import torch
    return {"cat": lambda xs, d: torch.cat(xs, dim=d),
            "full": lambda shape, v, like: torch.full(shape, v, dtype=like.dtype, device=like.device),
            "arange": lambda k, like: torch.arange(k, dtype=like.dtype, device=like.device)}


def suffix_ids(ref_tokenizer, suffix=None):
    s = os.environ.get("OPD_VLM_REF_SUFFIX", "<think>\n") if suffix is None else suffix
    return list(ref_tokenizer(s, add_special_tokens=False)["input_ids"]) if s else []


def build_batch(batch, ref_tokenizer):
    """The hook: the batch with ref_input_ids / ref_attention_mask / ref_position_ids from the student's own inputs."""
    b = batch.batch
    rm = b["response_mask"] if "response_mask" in b.keys() else b["attention_mask"][:, -b["responses"].shape[1]:]
    ids, mask, pos = build(b["input_ids"], b["attention_mask"], b["position_ids"], b["responses"], rm, suffix_ids(ref_tokenizer), torch_ops())
    b["ref_input_ids"], b["ref_attention_mask"], b["ref_position_ids"] = ids, mask, pos
    print(f"[opd_vlmref] ref inputs from the student's ids + {ids.shape[1] - b['input_ids'].shape[1]} suffix tokens: {tuple(ids.shape)}, positions {tuple(pos.shape)}")
    return batch
