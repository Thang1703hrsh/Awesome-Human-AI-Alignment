"""AlpacaEval 2 (Li et al., 2023; Dubois et al., 2024): win rate against GPT-4 Turbo on 805 instructions.

Two backends:

* ``official``: writes model outputs in the AlpacaEval format and calls the ``alpaca_eval`` package
  (``weighted_alpaca_eval_gpt4_turbo`` annotator). Use it for leaderboard-comparable raw and length-controlled (LC)
  win rates; the LC model needs the package's instruction-difficulty and gamed-baseline data.
* ``native``: any pairwise judge returning an AlpacaEval preference in [1, 2] (1 = baseline wins, 2 = model wins,
  1.5 = draw). Win rates follow ``AbsoluteScoringRule.describe_head2head`` (draws as 0 are mapped to 1.5, values
  outside [1, 2] are dropped) and ``ZeroOneScoringRule``. ``length_controlled_minimal`` fits the package's
  ``length_controlled_minimal`` formula, ``logit p = θ + φ·tanh(Δlen/sd(Δlen))`` with Δlen = len(baseline) −
  len(model) in characters, by unregularised maximum likelihood and predicts at Δlen = 0. It omits the instruction-
  difficulty term and the L1 cross-validation of the official fit, so it is a diagnostic, not a leaderboard number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Sequence

from human_alignment.benchmarks.common import load_hf_json, take

DATASET = "tatsu-lab/alpaca_eval"
EVAL_FILE = "alpaca_eval.json"
BASELINE_FILE = "alpaca_eval_gpt4_baseline.json"  # AlpacaEval 2 reference: gpt4_1106_preview outputs
DEFAULT_ANNOTATOR = "weighted_alpaca_eval_gpt4_turbo"

PreferenceJudge = Callable[[str, str, str], float | None]  # (instruction, baseline_output, model_output) -> [1, 2]


def load_alpacaeval(limit: int | None = None, revision: str | None = None) -> list[dict]:
    """The 805 evaluation instructions ``{instruction, dataset}``."""
    rows = [{"instruction": r["instruction"], "dataset": r["dataset"]} for r in load_hf_json(DATASET, EVAL_FILE, revision)]
    return take(rows, limit)


def load_alpacaeval_baseline(revision: str | None = None) -> dict[str, dict]:
    """Reference outputs keyed by instruction."""
    return {r["instruction"]: r for r in load_hf_json(DATASET, BASELINE_FILE, revision)}


def write_model_outputs(rows: Sequence[dict], path: str | Path, generator: str) -> Path:
    """Write ``[{instruction, output, generator, dataset}]``, the format ``alpaca_eval evaluate`` reads."""
    records = [
        {"instruction": r["instruction"], "output": r["output"], "generator": generator, "dataset": r.get("dataset")}
        for r in rows
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _valid(preferences: Sequence[float | None]) -> list[float]:
    values = []
    for p in preferences:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        p = 1.5 if p == 0 else float(p)
        if 1.0 <= p <= 2.0:
            values.append(p)
    return values


def win_rate(preferences: Sequence[float | None]) -> dict[str, float]:
    """Official head-to-head summary: weighted and discrete win rate (%), standard error, win/draw counts."""
    raw = [1.5 if p == 0 else p for p in preferences if p is not None]
    values = _valid(preferences)
    if not values:
        raise ValueError("no valid AlpacaEval preferences in [1, 2]")
    shifted = [p - 1 for p in values]
    discrete = [0.5 if p == 1.5 else float(round(p) == 2) for p in values]
    return {
        "win_rate": 100 * fmean(shifted),
        "standard_error": 100 * stdev(shifted) / math.sqrt(len(shifted)) if len(shifted) > 1 else float("nan"),
        "discrete_win_rate": 100 * fmean(discrete),
        "n_wins": sum(s > 0.5 for s in shifted),
        "n_wins_base": sum(s < 0.5 for s in shifted),
        "n_draws": sum(p == 1.5 for p in raw),
        "n_total": len(shifted),
        "n_dropped": len(preferences) - len(shifted),
    }


def length_controlled_minimal(
    preferences: Sequence[float | None], model_outputs: Sequence[str], baseline_outputs: Sequence[str]
) -> dict[str, float]:
    """Unregularised fit of AlpacaEval's ``length_controlled_minimal`` GLM, predicted at equal length."""
    triples = [
        (1.5 if p == 0 else float(p), len(b) - len(m))
        for p, m, b in zip(preferences, model_outputs, baseline_outputs)
        if p is not None and not (isinstance(p, float) and math.isnan(p)) and 1 <= (1.5 if p == 0 else p) <= 2
    ]
    if len(triples) < 3:
        raise ValueError("need at least three valid preferences")
    y = [p - 1 for p, _ in triples]
    deltas = [d for _, d in triples]
    sd = stdev(deltas) or 1.0
    x = [math.tanh(d / sd) for d in deltas]
    theta, phi = 0.0, 0.0
    for _ in range(100):  # Newton–Raphson on the soft-label logistic log-likelihood
        g0 = g1 = h00 = h01 = h11 = 0.0
        for yi, xi in zip(y, x):
            p = 1 / (1 + math.exp(-(theta + phi * xi)))
            r, w = yi - p, p * (1 - p)
            g0, g1 = g0 + r, g1 + r * xi
            h00, h01, h11 = h00 + w, h01 + w * xi, h11 + w * xi * xi
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (h00 * g1 - h01 * g0) / det
        theta, phi = theta + step0, phi + step1
        if max(abs(step0), abs(step1)) < 1e-10:
            break
    return {"lc_minimal_win_rate": 100 / (1 + math.exp(-theta)), "theta": theta, "length_coef": phi, "n_total": len(y)}


def evaluate_alpacaeval_native(
    rows: Sequence[dict], baseline: dict[str, dict], judge: PreferenceJudge
) -> tuple[list[dict], dict]:
    """Judge ``rows`` (``instruction``, ``output``) against the reference outputs with ``judge``."""
    judged = []
    for row in rows:
        ref = baseline[row["instruction"]]["output"]
        judged.append({**row, "baseline_output": ref, "preference": judge(row["instruction"], ref, row["output"])})
    prefs = [r["preference"] for r in judged]
    summary = {"benchmark": "AlpacaEval 2 (native judge)", **win_rate(prefs)}
    try:
        summary.update(
            length_controlled_minimal(prefs, [r["output"] for r in judged], [r["baseline_output"] for r in judged])
        )
    except ValueError:  # too few valid preferences to fit the length model
        summary["lc_minimal_win_rate"] = None
    return judged, summary


def evaluate_alpacaeval_official(
    model_outputs: str | Path, output_dir: str | Path, annotators_config: str = DEFAULT_ANNOTATOR, **kwargs: Any
) -> Any:
    """Run the ``alpaca_eval`` package (needs ``pip install alpaca-eval`` and its judge API key)."""
    try:
        from alpaca_eval import evaluate
    except ImportError as exc:
        raise ImportError("The official backend requires `pip install alpaca-eval`.") from exc
    return evaluate(
        model_outputs=str(model_outputs),
        annotators_config=annotators_config,
        output_path=str(output_dir),
        is_return_instead_of_print=True,
        **kwargs,
    )
