import re
from typing import List, Tuple


def parse_score(text: str):
    """
    Parse importance score from LLM output.

    Supported formats:

    Importance Score: 8
    Importance Score = 8
    Score: 8
    Score = 8
    8
    """

    patterns = [
        r"Importance\s*Score\s*[:=]\s*(\d+)",
        r"Score\s*[:=]\s*(\d+)",
        r"Importance\s*[:=]\s*(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            score = max(1, min(score, 10))
            return score

    # fallback:
    nums = re.findall(r"\b([1-9]|10)\b", text)

    if len(nums) > 0:
        return int(nums[0])

    return None


def parse_reason(text: str):
    """
    Parse reason.

    Supported formats:

    Reason:
    xxxx

    Reason = xxxx
    """

    patterns = [
        r"Reason\s*[:=]\s*(.*)",
    ]

    for pattern in patterns:

        m = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m:

            reason = m.group(1).strip()

            if len(reason) > 0:
                return reason

    return ""


def parse_output(text: str):
    """
    Return

    score
    reason
    """

    score = parse_score(text)
    reason = parse_reason(text)

    return score, reason


def aggregate_scores(scores: List[int]):
    """
    Average score for self-consistency.

    Ignore invalid outputs.
    """

    valid = []

    for s in scores:

        if s is None:
            continue

        valid.append(s)

    if len(valid) == 0:
        return None

    return sum(valid) / len(valid)


def majority_vote(scores: List[int]):
    """
    Majority vote.

    Example:

    [8,8,9,8,7]

    ->
    8
    """

    valid = []

    for s in scores:

        if s is not None:
            valid.append(s)

    if len(valid) == 0:
        return None

    counter = {}

    for s in valid:

        counter[s] = counter.get(s, 0) + 1

    counter = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return counter[0][0]


def aggregate_reasons(reasons: List[str]):
    """
    Keep the longest valid reason.

    Later can replace with
    LLM summarization.
    """

    valid = []

    for r in reasons:

        if r is None:
            continue

        r = r.strip()

        if len(r) == 0:
            continue

        valid.append(r)

    if len(valid) == 0:
        return ""

    valid = sorted(
        valid,
        key=lambda x: len(x),
        reverse=True,
    )

    return valid[0]


def self_consistency(outputs: List[str]):
    """
    outputs:

    [
        output1,
        output2,
        ...
    ]

    return

    {
        score,
        reason,
        all_scores,
        all_reasons
    }
    """

    scores = []
    reasons = []

    for out in outputs:

        score, reason = parse_output(out)

        scores.append(score)
        reasons.append(reason)

    result = {
        "score": aggregate_scores(scores),
        "majority_score": majority_vote(scores),
        "reason": aggregate_reasons(reasons),
        "all_scores": scores,
        "all_reasons": reasons,
    }

    return result


if __name__ == "__main__":

    example = """
    Importance Score: 8

    Reason:
    This behavior reflects the user's stable preference for science fiction rather than a temporary interest.
    """

    score, reason = parse_output(example)

    print(score)
    print(reason)

    outputs = [
        """
    Importance Score: 8
    Reason:
    Stable preference.
    """,
        """
    Importance Score: 9
    Reason:
    Long-term interest.
    """,
        """
    Score: 8
    Reason:
    Representative behavior.
    """,
    ]

    result = self_consistency(outputs)

    print(result)