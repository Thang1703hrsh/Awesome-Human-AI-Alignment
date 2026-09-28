"""Join candidate and baseline generations for win-rate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human_alignment.safety.eval.generate import read_jsonl


def attach_reference(
    candidate_path: str | Path, reference_path: str | Path, output_path: str | Path
) -> Path:
    candidate, reference = read_jsonl(candidate_path), read_jsonl(reference_path)
    if len(candidate) != len(reference):
        raise ValueError(
            f"generation lengths differ: {len(candidate)} != {len(reference)}"
        )
    rows = []
    for index, (cand, ref) in enumerate(zip(candidate, reference)):
        if cand.get("prompt") != ref.get("prompt"):
            raise ValueError(f"prompt mismatch at row {index}")
        rows.append(
            {
                **cand,
                "reference_response": ref["response"],
                "reference_model": ref.get("model"),
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    print(attach_reference(args.candidate, args.reference, args.output))


if __name__ == "__main__":
    main()

