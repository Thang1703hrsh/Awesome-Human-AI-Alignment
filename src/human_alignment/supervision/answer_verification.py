"""Programmatic answer checks used by PBSD's correctness gate.

Ported from ``verify_utils.py`` in the PBSD reference code
(github.com/LucasXinYu/PBSD), so "correct" means the same as in the authors'
training gate and GRPO reward. Math checking needs the optional ``math-verify``
package; the tool-use check is dependency-free.
"""

from __future__ import annotations

import os
import re

from human_alignment.exceptions import BackendUnavailableError


def extract_boxed_answer(text: str | None) -> str | None:
    """Return the content of the last ``\\boxed{...}`` after any final ``</think>``.

    Nested braces such as ``\\boxed{\\frac{1}{2}}`` are handled.
    """

    if not text:
        return None
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text
    index = search_text.find(r"\boxed{")
    if index == -1:
        return None
    start = index + len(r"\boxed{")
    depth = 1
    position = start
    while position < len(search_text) and depth > 0:
        if search_text[position] == "{":
            depth += 1
        elif search_text[position] == "}":
            depth -= 1
        position += 1
    if depth == 0:
        return search_text[start : position - 1].strip()
    return None


def _preprocess_for_parse(answer: str | None) -> str | None:
    """Convert ratio notation ``a:b`` to ``\\frac{a}{b}`` so math-verify parses it."""

    if answer is None:
        return None
    ratio = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio:
        return rf"\frac{{{ratio.group(1)}}}{{{ratio.group(2)}}}"
    return answer


def is_math_correct(completion: str, ground_truth: str | None) -> bool:
    """Symbolic match of the boxed answer, with a whitespace-insensitive string fallback."""

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise BackendUnavailableError(
            "PBSD's math correctness gate requires `pip install math-verify`."
        ) from exc
    # math-verify's default 5 s timeouts use SIGALRM on POSIX, as in the reference
    # runs; its Windows fallback spawns a process that cannot pickle the parser and
    # silently returns no parse, so timeouts are disabled there.
    timeouts = ({} if os.name == "posix" else {"parsing_timeout": None})
    verify_timeouts = ({} if os.name == "posix" else {"timeout_seconds": None})
    predicted = extract_boxed_answer(completion)
    gold_parsed = parse(ground_truth, **timeouts) if ground_truth is not None else None
    predicted_parsed = parse(_preprocess_for_parse(predicted), **timeouts)
    if gold_parsed is not None and predicted_parsed is not None:
        try:
            if verify(gold_parsed, predicted_parsed, **verify_timeouts):
                return True
        except Exception:
            # math-verify raises on unparsable or unsupported expressions; fall back below.
            pass
    predicted_norm = re.sub(r"\s+", "", predicted or "").lower()
    gold_norm = re.sub(r"\s+", "", ground_truth or "").lower()
    return bool(predicted_norm) and predicted_norm == gold_norm


def _action_part(text: str | None) -> str:
    if not text:
        return ""
    index = text.find("Action:")
    return text[index:].strip() if index != -1 else text.strip()


def is_tooluse_correct(completion: str, ground_truth: str | None) -> bool:
    """True iff the completion's ``Action:`` section contains the gold action."""

    predicted = re.sub(r"\s+", "", _action_part(completion)).lower()
    gold = re.sub(r"\s+", "", _action_part(ground_truth)).lower()
    return bool(gold) and gold in predicted


def is_correct(completion: str, ground_truth: str | None, task: str = "math") -> bool:
    if task == "tooluse":
        return is_tooluse_correct(completion, ground_truth)
    if task == "math":
        return is_math_correct(completion, ground_truth)
    raise ValueError(f"Unknown verification task: {task}")
