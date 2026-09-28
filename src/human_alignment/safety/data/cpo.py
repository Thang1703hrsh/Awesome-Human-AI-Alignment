"""CPO (Controllable Preference Optimization, arXiv 2402.19085) data construction.

All CPO-specific logic is in the data; training is plain SFT (CPSFT) and plain DPO (CDPO).
Ports keep the authors' control flow, ``and``/``or`` idioms and the order of ``random.choice`` calls so that the
same seed produces the same pairs (verified in tests against the original ``Start``).
"""

from __future__ import annotations

import json
import random
from typing import Literal

DATA_KEYS = {
    "Help": ["Helpful_Rating", "Helpfulness_Rating"],
    "Honesty": ["Honesty_Rating"],
    "Harmless": ["Harmlessness_Rating"],
}

Variant = Literal["ultrasafety", "ultrafeedback"]


def _rate(response: dict, keys: list[str]):
    for key in keys:
        if response.get(key):
            return int(response.get(key))
    return None


def _modify_instruction(
    help_rating, honesty_rating, harmless_rating, result: dict
) -> None:
    text = ""
    if help_rating is not None:
        text = (
            text + f"< helplessness: {help_rating} > "
        )  # sic: the original tag says "helplessness"
    if honesty_rating is not None:
        text = text + f"< honesty: {honesty_rating} > "
    if harmless_rating is not None:
        text = text + f"< harmlessness: {harmless_rating} > "
    result["instruction"] = text + result["instruction"]


def _sample_target_responses(responses: list[dict], cfg: dict) -> list[dict]:
    output = []
    for r in responses:
        valid = True
        for cfg_key in cfg:
            if cfg_key not in DATA_KEYS:
                continue
            rate = cfg.get(cfg_key)
            if rate is None:
                continue
            if _rate(r, DATA_KEYS[cfg_key]) != rate:
                valid = False
                break
        if valid:
            output.append(r)
    return output


def build_cdpo_pairs(
    data: list[dict],
    has_harmless: bool,
    random_cfg: list[dict],
    variant: Variant,
    rng: random.Random | None = None,
) -> list[dict]:
    """Port of ``Start`` in ``CPO/src/CDPO/data_preparation/data_dpo_{ultrasafety,ultrafeedback}.py``."""
    rng = rng or random
    results = []
    for CFG in random_cfg:
        r1_enable = CFG["r1_enable"]
        r2_enable = CFG["r2_enable"]
        for cfg in CFG["random_range"].values():
            max_count = cfg["max_count"]
            count = 0
            for item in data:
                if count >= max_count:
                    break
                samples = _sample_target_responses(item["responses"], cfg)
                if not samples:
                    continue
                sample = rng.choice(samples)

                if variant == "ultrasafety":

                    def get_r(
                        response,
                        sample=sample,
                        r1_enable=r1_enable,
                        r2_enable=r2_enable,
                    ):
                        R3 = (
                            has_harmless
                            and (
                                -abs(
                                    _rate(response, DATA_KEYS["Harmless"])
                                    - int(_rate(sample, DATA_KEYS["Harmless"]))
                                )
                            )
                            or 0
                        )
                        R1 = r1_enable and _rate(response, DATA_KEYS["Help"]) or 0
                        R2 = r2_enable and _rate(response, DATA_KEYS["Honesty"]) or 0
                        return R3 + R1 + R2
                elif variant == "ultrafeedback":

                    def get_r(
                        response,
                        sample=sample,
                        r1_enable=r1_enable,
                        r2_enable=r2_enable,
                    ):
                        R3 = (
                            has_harmless
                            and (
                                -abs(
                                    _rate(response, DATA_KEYS["Harmless"])
                                    - int(_rate(sample, DATA_KEYS["Harmless"]))
                                )
                            )
                            or 0
                        )
                        R1 = (
                            r1_enable
                            and -abs(
                                _rate(response, DATA_KEYS["Help"])
                                - int(_rate(sample, DATA_KEYS["Help"]))
                            )
                            or int(_rate(response, DATA_KEYS["Help"]))
                        )
                        # sic: -abs(x) - y, not -abs(x - y); falls back to x when that is 0.
                        R2 = (
                            r2_enable
                            and -abs(_rate(response, DATA_KEYS["Honesty"]))
                            - int(_rate(sample, DATA_KEYS["Honesty"]))
                            or _rate(response, DATA_KEYS["Honesty"])
                        )
                        return R3 + R1 + R2
                else:
                    raise ValueError(f"Unknown CPO variant {variant!r}")

                instruction = item["instruction"]
                responses = item["responses"]
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        if count >= max_count:
                            break
                        R_i, R_j = get_r(responses[i]), get_r(responses[j])
                        if R_i > R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": responses[i]["response"],
                                "reject": responses[j]["response"],
                            }
                        elif R_i < R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": responses[j]["response"],
                                "reject": responses[i]["response"],
                            }
                            R_i, R_j = R_j, R_i
                        else:
                            continue
                        result["R_chosen"] = R_i
                        result["R_reject"] = R_j
                        _modify_instruction(
                            r1_enable and _rate(sample, DATA_KEYS["Help"]) or None,
                            r2_enable and _rate(sample, DATA_KEYS["Honesty"]) or None,
                            _rate(sample, DATA_KEYS["Harmless"]),
                            result,
                        )
                        results.append(result)
                        count = count + 1
    return results


def further_process(data: list[dict]) -> list[dict]:
    """``further_process_data`` (``cdpo_ultrasafety.py:170-181``): drop scores, strip the default-safe tag, rename."""
    out = []
    for item in data:
        item = dict(item)
        item.pop("R_chosen", None)
        item.pop("R_reject", None)
        if "instruction" in item:
            item["instruction"] = item["instruction"].replace(
                "< harmlessness: 1 > ", ""
            )
            item["prompt"] = item.pop("instruction")
        if "reject" in item:
            item["rejected"] = item.pop("reject")
        out.append(item)
    return out


def cpsft_examples(jsonl_lines: list[str]) -> list[dict]:
    """``data_preparation_cpsft.py`` (active branch): ``"< honesty: R > " + instruction`` per completion."""
    results = []
    for line in jsonl_lines:
        d = json.loads(line)
        for completion in d["completions"]:
            honesty = (
                "< honesty: " + completion["annotations"]["honesty"]["Rating"] + " >"
            )
            results.append(
                {
                    "instruction": honesty + " " + d["instruction"],
                    "input": "",
                    "output": completion["response"],
                }
            )
    return results


def _annotation_rating(annotation) -> str:
    """Read both the released HF schema (a one-element list) and the authors' flattened shards."""
    if isinstance(annotation, list):
        annotation = annotation[0] if annotation else {}
    if not isinstance(annotation, dict) or "Rating" not in annotation:
        raise ValueError("CPO annotation has no Rating")
    return str(annotation["Rating"])


def cpsft_examples_from_rows(rows: list[dict]) -> list[dict]:
    """CPSFT examples directly from ``openbmb/UltraFeedback`` rows.

    This removes the undocumented manual step of downloading six JSONL shards;
    it preserves the active author branch: honesty control only, all completions.
    """
    results = []
    for row in rows:
        for completion in row["completions"]:
            rating = _annotation_rating(completion["annotations"]["honesty"])
            results.append(
                {
                    "instruction": f"< honesty: {rating} > {row['instruction']}",
                    "input": "",
                    "output": completion["response"],
                }
            )
    return results


def flatten_annotated_rows(rows: list[dict]) -> list[dict]:
    """Convert released UltraSafety/UltraFeedback rows into the CDPO author's intermediate schema."""
    out = []
    for row in rows:
        responses = []
        for completion in row["completions"]:
            ann = completion["annotations"]
            responses.append(
                {
                    "response": completion["response"],
                    "Harmlessness_Rating": _annotation_rating(ann["harmlessness"]),
                    "Helpfulness_Rating": _annotation_rating(ann["helpfulness"]),
                    "Honesty_Rating": _annotation_rating(ann["honesty"]),
                }
            )
        out.append({"instruction": row["instruction"], "responses": responses})
    return out


def cpsft_text(example: dict) -> str:
    """``mistral_delete`` template (``prompt_no_input = "{instruction}"``) + label appended with no separator.
    CPSFT trains on the whole text (``train_on_inputs=True``) and appends EOS."""
    if example.get("input"):
        return f"{example['instruction']}\n{example['input']}{example['output']}"
    return f"{example['instruction']}{example['output']}"


ULTRASAFETY_CFG = {
    "has_harmless": True,
    "random_cfg": [
        {
            "r1_enable": False,
            "r2_enable": False,
            "random_range": {
                "1": {"max_count": 8966, "Harmless": 1},
                "0": {"max_count": 8966, "Harmless": 0},
            },
        }
    ],
}

