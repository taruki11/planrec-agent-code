#!/usr/bin/env python3
"""Full-catalog constrained semantic-ID generation evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.genrec.constraints import TokenSequenceTrie
from planrec.genrec.hf_runtime import load_model_and_tokenizer
from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    args = parser.parse_args()

    semantic_ids = ResidualKMeansSemanticTokenizer.load(args.semantic_ids)
    model, tokenizer = load_model_and_tokenizer(args.model, semantic_ids, args.dtype, args.attention)
    model.to(torch.device(args.device)).eval()
    rows = read_jsonl(args.test_file, args.limit)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("model tokenizer must define eos_token_id")
    sequences: list[tuple[list[int], int]] = []
    for item_id in sorted(semantic_ids.item_to_code):
        tokens = semantic_ids.completion_tokens_for_item(item_id)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
            raise RuntimeError("semantic token registration failed")
        sequences.append(([*token_ids, eos], item_id))
    trie = TokenSequenceTrie.from_sequences(sequences)
    completion_length = len(sequences[0][0])

    hits = 0
    ndcg = 0.0
    invalid = 0
    predicted = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [str(row["prompt"]) for row in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]

            def allowed_tokens(batch_id: int, input_ids: torch.Tensor) -> list[int]:
                del batch_id
                generated = input_ids[prompt_width:].tolist()
                allowed = trie.allowed_next(generated)
                return [int(token) for token in allowed] if allowed else [eos]

            outputs = model.generate(
                **encoded,
                max_new_tokens=completion_length,
                num_beams=args.k,
                num_return_sequences=args.k,
                do_sample=False,
                early_stopping=True,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,
            )
            generated = outputs[:, prompt_width:].cpu().tolist()
            for row_index, row in enumerate(batch):
                target = int(row["target_item"])
                ranked: list[int] = []
                for sequence in generated[row_index * args.k : (row_index + 1) * args.k]:
                    item = trie.decode(sequence)
                    predicted += 1
                    if item is None:
                        invalid += 1
                    elif item not in ranked:
                        ranked.append(item)
                if target in ranked:
                    rank = ranked.index(target) + 1
                    hits += 1
                    ndcg += 1.0 / math.log2(rank + 1)

    total = len(rows)
    result = {
        "protocol": "full_catalog_constrained_sid_beam_v1",
        "num_examples": total,
        "beam_k": args.k,
        "hr@k": hits / max(total, 1),
        "ndcg@k": ndcg / max(total, 1),
        "invalid_rate": invalid / max(predicted, 1),
        "model": args.model,
        "semantic_ids": str(args.semantic_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
