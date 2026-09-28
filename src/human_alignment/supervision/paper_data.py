"""Paper-specific feedback ingestion; no model downloads or training side effects.

Dataset adapters preserve annotation provenance. They do not reproduce the
training algorithms of the papers. See docs/ALIGNMENT_SUPERVISION.md.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping

from human_alignment.exceptions import DatasetFormatError
from human_alignment.supervision.signals import Demonstration, Preference
from human_alignment.types import (
    FeedbackRepresentation as Representation,
    FeedbackSource as Source,
    InteractionContext,
    SupervisionSignal,
)


def number(value: Any, name: str, low: float | None = None,
           high: float | None = None) -> float:
    if isinstance(value, bool):
        raise DatasetFormatError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (low is not None and result < low) or (
        high is not None and result > high
    ):
        raise DatasetFormatError(f"{name} is nonfinite or outside its allowed range")
    return result


def _pair(prompt, chosen, rejected, source, paper, metadata=None):
    for response in (chosen, rejected):
        if response is None or (isinstance(response, str) and not response.strip()) or (
            isinstance(response, (list, tuple)) and not response
        ):
            raise DatasetFormatError("Preference responses must be nonempty")
    if chosen == rejected:
        raise DatasetFormatError("Identical responses cannot form a strict preference")
    return SupervisionSignal(
        InteractionContext(prompt), source, Representation.PREFERENCE,
        Preference((chosen, rejected), 0),
        provenance={**dict(metadata or {}), "paper": paper},
    )


def _ranked(prompt, responses, scores, source, paper, metadata=None):
    if isinstance(responses, (str, bytes)) or len(responses) < 2:
        raise DatasetFormatError("At least two responses are required")
    if len(responses) != len(scores):
        raise DatasetFormatError("Response and score counts differ")
    scores = [number(score, "score") for score in scores]
    result = []
    for i, j in combinations(range(len(responses)), 2):
        if scores[i] == scores[j] or responses[i] == responses[j]:
            continue  # Ties are never silently converted into strict preferences.
        winner, loser = (i, j) if scores[i] > scores[j] else (j, i)
        result.append(_pair(prompt, responses[winner], responses[loser], source, paper,
                            {**dict(metadata or {}), "indices": [winner, loser],
                             "scores": [scores[winner], scores[loser]]}))
    return result


class PaperFeedbackDataset:
    """Normalize records once, or use as an AlignmentPipeline feedback provider.

    ``collect`` accepts empty contexts for the whole supplied dataset. Nonempty
    contexts are rejected to avoid silently ignoring a pipeline's input selection.
    """

    FORMATS = {
        "human_preferences", "summarization_feedback", "instructgpt_feedback",
        "hh_rlhf", "openassistant", "helpsteer2", "fine_grained_feedback",
        "rlhf_v", "mm_rlhf", "wildfeedback", "ultrafeedback",
    }

    def __init__(self, format: str, records: Iterable[Mapping[str, Any]], *,
                 annotation_source: Source = Source.HUMAN):
        if format not in self.FORMATS:
            raise DatasetFormatError(f"Unsupported feedback format: {format}")
        self.format = self.name = format
        self.records = tuple(records)
        if any(not isinstance(row, Mapping) for row in self.records):
            raise DatasetFormatError("Feedback records must be mappings")
        self.annotation_source = Source(annotation_source)

    def collect(self, contexts, specification):
        if contexts:
            raise DatasetFormatError("Dataset providers require contexts=(); prefilter records")
        return self.convert()

    def convert(self) -> list[SupervisionSignal]:
        if self.format == "openassistant":
            return self._openassistant()
        if self.format == "helpsteer2":
            return self._helpsteer2()
        output = []
        for index, row in enumerate(self.records):
            try:
                output.extend(self._row(row))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise DatasetFormatError(f"{self.format} row {index}: {exc}") from exc
        return output

    def _row(self, row):
        name = self.format
        source = self.annotation_source
        metadata = dict(row.get("metadata", {}))
        if name == "summarization_feedback":
            info = row["info"]
            text = info.get("post", info.get("article"))
            if not isinstance(text, str) or not text:
                raise DatasetFormatError("info.post or info.article is required")
            summaries = row["summaries"]
            choice = row["choice"]
            if len(summaries) != 2 or type(choice) is not int or choice not in (0, 1):
                raise DatasetFormatError("Expected two summaries and integer choice 0 or 1")
            return [_pair(text, summaries[choice]["text"], summaries[1-choice]["text"],
                          source, name, {**metadata, "info": info, "split": row.get("split"),
                                         "batch": row.get("batch")})]
        if name == "hh_rlhf":
            chosen, rejected = row["chosen"], row["rejected"]
            marker = "\n\nAssistant:"
            cp, cm, cc = chosen.rpartition(marker)
            rp, rm, rc = rejected.rpartition(marker)
            if not cm or not rm or cp != rp:
                raise DatasetFormatError("HH transcripts must share the same final-turn prompt")
            return [_pair(cp + marker, cc, rc, source, name, metadata)]
        if name == "instructgpt_feedback":
            # Public interface for user-provided demonstrations/rankings; not a
            # claim that OpenAI's original private training dataset is available.
            if "completion" in row:
                return [SupervisionSignal(InteractionContext(row["prompt"]), source,
                        Representation.DEMONSTRATION, Demonstration(row["completion"]),
                        provenance={**metadata, "paper": name})]
            ranks = [number(v, "rank", 0) for v in row["ranks"]]
            return _ranked(row["prompt"], row["responses"], [-v for v in ranks],
                           source, name, metadata)
        if name == "human_preferences":
            options = row["segments"]
            probability = number(row["preference_probability"], "preference_probability", 0, 1)
            if len(options) != 2 or not all(isinstance(x, (list, tuple)) and x for x in options):
                raise DatasetFormatError("Expected two nonempty trajectory segments")
            # Preserve soft labels/ties for Bradley-Terry reward learning.
            return [SupervisionSignal(InteractionContext(row.get("prompt", "trajectory")),
                    source, Representation.TRAJECTORY,
                    {"segments": options, "preference_probability": probability},
                    provenance={**metadata, "paper": name})]
        if name == "fine_grained_feedback":
            return [_pair(row["prompt"], row["corrected"], row["original"], source, name,
                          {**metadata, "annotation": "minimal_edit"})]
        if name == "rlhf_v":
            text = row["text"]
            data = json.loads(text) if isinstance(text, str) else text
            image = row.get("image")
            if image is None:
                image = row.get("image_path")
            if image is None:
                raise DatasetFormatError("RLHF-V requires image or image_path")
            return [_pair({"text": data["question"], "image": image},
                          data["chosen"], data["rejected"], source, name,
                          {**metadata, "image_path": row.get("image_path"),
                           "origin_dataset": row.get("origin_dataset"),
                           "annotation": "minimal_edit"})]
        if name == "mm_rlhf":
            media = {key: row[key] for key in ("image", "video") if row.get(key) is not None}
            if not media:
                raise DatasetFormatError("MM-RLHF requires image or video")
            ranks = [number(v, "rank", 1) for v in row["final_ranking"]]
            return _ranked({"text": row["question"], **media}, row["models_output"],
                           [-v for v in ranks], source, name,
                           {**metadata, "score_reasons": row.get("score_reasons"),
                            "ranking_reason": row.get("ranking_reason"),
                            "ratings": {k: row.get(k) for k in
                                        ("faithfulness", "helpfulness", "ethical")}})
        if name == "wildfeedback":
            # Explicit normalized schema; raw SAT/DSAT logs are NOT preference pairs.
            return [_pair(row["prompt"], row["chosen"], row["rejected"], Source.HYBRID, name,
                          {**metadata, "user_feedback": row["user_feedback"],
                           "annotation": "in_situ_human_feedback_with_ai_processing"})]
        if name == "ultrafeedback":
            completions = row["completions"]
            scores = []
            for completion in completions:
                ratings = completion["annotations"]
                values = [number(ratings[k]["Rating"], k, 1, 5) for k in
                          ("instruction_following", "truthfulness", "honesty", "helpfulness")]
                scores.append(sum(values) / len(values))
            return _ranked(row["instruction"], [x["response"] for x in completions], scores,
                           Source.AI, name, {**metadata, "completions": completions,
                                             "aggregation": "mean_four_aspects"})
        raise DatasetFormatError(f"No row converter for {name}")

    def _helpsteer2(self):
        groups = defaultdict(list)
        for row in self.records:
            ratings = {key: number(row[key], key, 0, 4) for key in
                       ("helpfulness", "correctness", "coherence", "complexity", "verbosity")}
            groups[row["prompt"]].append((row["response"], ratings))
        result = []
        for prompt, rows in groups.items():
            if len(rows) < 2:
                continue
            result.extend(_ranked(prompt, [r[0] for r in rows],
                                  [r[1]["helpfulness"] for r in rows], Source.HUMAN,
                                  self.name, {"ratings": [r[1] for r in rows],
                                              "aggregation": "helpfulness_only"}))
        return result

    def _openassistant(self):
        nodes = {row["message_id"]: row for row in self.records}
        if len(nodes) != len(self.records):
            raise DatasetFormatError("Duplicate OpenAssistant message IDs")
        siblings = defaultdict(list)
        for row in self.records:
            if row["role"] == "assistant" and row.get("rank") is not None:
                if not row.get("deleted", False) and row.get("review_result") is not False:
                    siblings[row["parent_id"]].append(row)
        output = []
        for parent, replies in siblings.items():
            history, seen, current = [], set(), parent
            while current is not None:
                if current in seen or current not in nodes:
                    raise DatasetFormatError("OpenAssistant ancestry is cyclic or incomplete")
                seen.add(current)
                node = nodes[current]
                if node.get("deleted", False) or node.get("review_result") is False:
                    history = []
                    break
                role = {"prompter": "user", "assistant": "assistant"}.get(node["role"])
                if role is None:
                    raise DatasetFormatError("Unknown OpenAssistant role")
                history.append({"role": role, "content": node["text"]})
                current = node.get("parent_id")
            if not history or len(replies) < 2:
                continue
            output.extend(_ranked(list(reversed(history)),
                [[{"role": "assistant", "content": r["text"]}] for r in replies],
                [-number(r["rank"], "rank", 0) for r in replies], Source.HUMAN,
                self.name, {"parent_id": parent,
                            "message_ids": [r["message_id"] for r in replies]}))
        return output
