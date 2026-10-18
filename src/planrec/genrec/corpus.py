from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


class GenRecCorpusBuilder:
    """Build pretraining, next-item SFT, and offline preference records."""

    def __init__(self, semantic_ids: ResidualKMeansSemanticTokenizer, max_history: int = 50) -> None:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.semantic_ids = semantic_ids
        self.max_history = int(max_history)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for item_id, code in semantic_ids.item_to_code.items():
            for length in range(semantic_ids.num_levels + 1):
                buckets.setdefault(code[:length], []).append(item_id)
        self._prefix_items = {prefix: sorted(items) for prefix, items in buckets.items()}

    def render_item(self, item_id: int) -> str:
        tokens = ["<item>", *self.semantic_ids.tokens_for_item(item_id), "</item>"]
        return " ".join(tokens)

    def prompt(self, history: Iterable[int]) -> str:
        clean_history = [int(item) for item in history if int(item) > 0][-self.max_history :]
        items = " ".join(self.render_item(item) for item in clean_history)
        return f"<rec_history> {items} </rec_history> <rec_target>"

    def completion(self, item_id: int) -> str:
        return f"{self.render_item(item_id)} </rec_target>"

    def sft_record(self, history: Iterable[int], target_item: int) -> dict[str, object]:
        return {
            "prompt": self.prompt(history),
            "completion": self.completion(target_item),
            "target_item": int(target_item),
        }

    def pretrain_record(self, history: Iterable[int], target_item: int) -> dict[str, str]:
        record = self.sft_record(history, target_item)
        return {"text": f"{record['prompt']} {record['completion']}"}

    def hard_negative(self, target_item: int, banned_items: Iterable[int] = ()) -> int:
        banned = {int(item) for item in banned_items}
        banned.add(int(target_item))
        target_code = self.semantic_ids.item_to_code[int(target_item)]
        for length in range(self.semantic_ids.num_levels, -1, -1):
            for item_id in self._prefix_items.get(target_code[:length], []):
                if item_id not in banned:
                    return item_id
        raise ValueError("no valid negative item remains after filtering")

    def preference_record(
        self,
        history: Iterable[int],
        target_item: int,
        rejected_item: int | None = None,
    ) -> dict[str, object]:
        history_list = [int(item) for item in history if int(item) > 0]
        rejected = rejected_item or self.hard_negative(target_item, history_list)
        if rejected == target_item:
            raise ValueError("chosen and rejected items must differ")
        return {
            "prompt": self.prompt(history_list),
            "chosen": self.completion(target_item),
            "rejected": self.completion(rejected),
            "target_item": int(target_item),
            "rejected_item": int(rejected),
        }

    def alignment_records(self, item_texts: Mapping[int, str]) -> Iterator[dict[str, str]]:
        for item_id in sorted(item_texts):
            text = str(item_texts[item_id]).strip()
            if not text or item_id not in self.semantic_ids.item_to_code:
                continue
            item = self.render_item(item_id)
            yield {"prompt": f"Map this item text to its item token: {text}\nItem token:", "completion": item}
            yield {"prompt": f"Describe the product represented by {item}\nDescription:", "completion": text}


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
