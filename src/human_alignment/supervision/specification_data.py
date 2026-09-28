"""Task-mixture, multi-turn and individual-feedback data preparation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import combinations

from human_alignment.supervision.datasets import InstructionExample, PreferenceExample


class FlanMixture:
    """Sample task collections using explicit mixture weights.

    Accepts FLAN-style inputs/targets. Weights are required: equal/default weights
    must not be mistaken for the published FLAN mixture recipe.
    """

    def __init__(self, collections, weights, *, seed=0):
        if not collections or set(collections) != set(weights):
            raise ValueError("Specify one weight for each nonempty task collection")
        self.collections = {key: tuple(rows) for key, rows in collections.items()}
        self.weights = {key: float(value) for key, value in weights.items()}
        if any(not rows for rows in self.collections.values()) or any(
            not math.isfinite(w) or w <= 0 for w in self.weights.values()
        ):
            raise ValueError("Collections and finite weights must be positive")
        self.seed = seed

    def sample(self, count):
        if type(count) is not int or count <= 0:
            raise ValueError("count must be positive")
        rng, names = random.Random(self.seed), sorted(self.collections)
        result = []
        for _ in range(count):
            name = rng.choices(names, weights=[self.weights[n] for n in names], k=1)[0]
            row = rng.choice(self.collections[name])
            result.append(InstructionExample(row["inputs"], row["targets"],
                {"collection": name, "mixture_weight": self.weights[name],
                 "source_metadata": row.get("metadata", {})}))
        return result


class ParrotTurns:
    """Turn-wise SFT export retaining full preceding instruction history.

    Input is normalized messages, not a parser of every upstream Parrot release.
    The paper's synthetic tree construction is not reproduced by this adapter.
    """

    def convert(self, conversations):
        output = []
        for conversation in conversations:
            history = []
            for index, message in enumerate(conversation["messages"]):
                role = message["role"]
                if role not in {"system", "user", "assistant"}:
                    raise ValueError("Only system/user/assistant messages are supported")
                if not isinstance(message["content"], str) or not message["content"].strip():
                    raise ValueError("Message content must be nonempty text")
                if role == "assistant":
                    if not history or history[-1]["role"] != "user":
                        raise ValueError("An assistant training target must follow a user turn")
                    output.append(InstructionExample(list(history), [dict(message)],
                        {"conversation_id": conversation.get("id"), "turn": index}))
                history.append(dict(message))
        return output


class PRISMPreferences:
    """Official conversation_history adapter, paired within user/conversation/turn.

    Only the selected branch is retained in subsequent context. Missing selection
    in a nonfinal turn is rejected rather than guessing a conversation path.
    """

    def convert(self, conversations):
        output = []
        for row in conversations:
            user = row["user_id"]
            if not isinstance(user, str) or not user:
                raise ValueError("PRISM records require user_id")
            turns = defaultdict(list)
            for message in row["conversation_history"]:
                turns[message["turn"]].append(message)
            history = []
            ordered_turns = sorted(turns)
            for turn in ordered_turns:
                messages = turns[turn]
                prompts = [m for m in messages if m["role"] == "user"]
                responses = [m for m in messages if m["role"] == "model"]
                if len(prompts) != 1 or not responses:
                    raise ValueError("Each PRISM turn requires one user message and model responses")
                history.append({"role": "user", "content": prompts[0]["content"]})
                rated = [m for m in responses if m.get("score") is not None]
                for a, b in combinations(rated, 2):
                    sa, sb = float(a["score"]), float(b["score"])
                    if not all(math.isfinite(s) and 0 <= s <= 100 for s in (sa, sb)):
                        raise ValueError("PRISM ratings must be finite in [0, 100]")
                    if sa == sb or a["content"] == b["content"]:
                        continue
                    chosen, rejected = (a, b) if sa > sb else (b, a)
                    output.append(PreferenceExample(list(history),
                        [{"role": "assistant", "content": chosen["content"]}],
                        [{"role": "assistant", "content": rejected["content"]}],
                        {"user_id": user, "conversation_id": row["conversation_id"],
                         "turn": turn, "source": "human", "paper": "prism",
                         "chosen_model": chosen.get("model_name"),
                         "rejected_model": rejected.get("model_name")}))
                selected = [m for m in responses if m.get("if_chosen") is True]
                if len(selected) > 1 or (turn != ordered_turns[-1] and len(selected) != 1):
                    raise ValueError("Cannot reconstruct PRISM selected history")
                if selected:
                    history.append({"role": "assistant", "content": selected[0]["content"]})
        return output
