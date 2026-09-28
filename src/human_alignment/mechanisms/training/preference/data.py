"""Pair collation with explicit token-weight alignment for native objectives."""

import torch


class PreferenceCollator:
    """Accept text pairs or explicitly tokenized pairs.

    Text weights cover completion tokens *before* EOS is appended. Pretokenized
    weights cover input_ids including masked prompt positions. They are final
    nonnegative importance weights, not upstream signed contrastive scores.
    """

    def __init__(self, tokenizer, max_length=1024, max_prompt_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def encode(self, row):
        result = {}
        if "chosen_input_ids" in row:
            for side in ("chosen", "rejected"):
                ids = list(row[f"{side}_input_ids"])
                labels = list(row[f"{side}_labels"])
                if len(ids) != len(labels) or len(ids) > self.max_length:
                    raise ValueError("Pretokenized IDs/labels must align and fit max_length")
                if any(label != -100 and label != token for label, token in zip(labels, ids)):
                    raise ValueError("Unmasked labels must equal their input token IDs")
                valid = [i for i, label in enumerate(labels) if label != -100]
                if not valid or valid[0] == 0 or valid != list(range(valid[0], len(ids))):
                    raise ValueError("Require a nonempty masked prompt followed by contiguous response labels")
                result[f"{side}_input_ids"] = ids
                result[f"{side}_labels"] = labels
                if f"{side}_weights" in row:
                    weights = list(row[f"{side}_weights"])
                    if len(weights) != len(ids):
                        raise ValueError("Pretokenized weights must match input_ids exactly")
                    result[f"{side}_weights"] = weights
            return result
        prompt = row["prompt"]
        if not all(isinstance(row[key], str) for key in ("prompt", "chosen", "rejected")):
            raise ValueError("Native preference methods require rendered text strings or tokenized pairs")
        tokenizer = self.tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id")
        pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if tokenizer.bos_token_id is not None and (not pids or pids[0] != tokenizer.bos_token_id):
            pids = [tokenizer.bos_token_id] + pids
        if not pids:
            raise ValueError("A nonempty prompt or BOS token is required")
        pids = pids[-self.max_prompt_length:]
        for side in ("chosen", "rejected"):
            response = tokenizer(row[side], add_special_tokens=False)["input_ids"]
            weights = row.get(f"{side}_weights")
            if weights is not None and len(weights) != len(response):
                raise ValueError(f"{side}_weights must match completion tokens before appended EOS")
            room = self.max_length - len(pids)
            ids = (response + [tokenizer.eos_token_id])[:room]
            result[f"{side}_input_ids"] = pids + ids
            result[f"{side}_labels"] = [-100] * len(pids) + ids
            if weights is not None:
                result[f"{side}_weights"] = [0.] * len(pids) + (list(weights) + [0.])[:room]
        return result

    def __call__(self, rows):
        features = [self.encode(row) for row in rows]
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("A pad or EOS token is required")
        size = max(len(row[f"{s}_input_ids"]) for row in features for s in ("chosen", "rejected"))
        result = {}
        presence = [f"{s}_weights" in row for s in ("chosen", "rejected") for row in features]
        if any(presence) and not all(presence):
            raise ValueError("Provide weights for both sides of every pair, or none")
        for side in ("chosen", "rejected"):
            ids, labels, masks, weights = [], [], [], []
            for row in features:
                seq = row[f"{side}_input_ids"]
                padding = size - len(seq)
                ids.append(seq + [pad] * padding)
                labels.append(row[f"{side}_labels"] + [-100] * padding)
                masks.append([1] * len(seq) + [0] * padding)
                if all(presence):
                    weights.append(row[f"{side}_weights"] + [0.] * padding)
            for key, value in (("input_ids", ids), ("labels", labels), ("attention_mask", masks)):
                result[f"{side}_{key}"] = torch.tensor(value, dtype=torch.long)
            if weights:
                result[f"{side}_weights"] = torch.tensor(weights, dtype=torch.float32)
        return result
