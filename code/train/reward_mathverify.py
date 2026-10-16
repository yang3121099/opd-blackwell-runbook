"""verl custom reward: 1.0 if the LAST \\boxed{} answer matches ground truth under Math-Verify, else 0.0.
Responses without a boxed answer get 0 (R-001 item 6: grading the whole text turns multiple boxed
answers into a set and penalises answer revisions)."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    from math_verify import parse, verify
    i = solution_str.rfind("\\boxed{")
    if i < 0:
        return 0.0
    try:
        pred = parse(solution_str[i:])
        if not pred:
            return 0.0
        gold = parse(f"${ground_truth}$")
        return 1.0 if verify(gold, pred) else 0.0
    except Exception:
        return 0.0
