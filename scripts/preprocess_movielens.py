#!/usr/bin/env python3
"""MovieLens-1M -> frozen chronological LOO dataset (PlanRec E2 protocol).

- sort by (user_id, timestamp); drop duplicate (user,item) keep last rating
- per user: last rating = test, second-to-last = validation, everything before
  = train (prefix-samples: every prefix ending at a train rating is a sample)
- right padding, max_seq_len=512, item ids remapped to 1..N (row 0 = padding)
- output data/movielens_1m/processed/datasets.pt + data_stats.json
- item texts: title + genres for the frozen-encoder item embeddings later
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "movielens_1m"
RAW = OUT / "ml-1m"
MAX_SEQ = 512

def load_ratings() -> list[tuple[int, int, int]]:
    rows = []
    with (RAW / "ratings.dat").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[3])))
    return rows

def load_item_texts() -> dict[int, str]:
    texts = {}
    with (RAW / "movies.dat").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            texts[int(parts[0])] = f"{parts[1]} {parts[2]}"
    return texts

def main() -> None:
    rows = load_ratings()
    print(f"raw interactions: {len(rows)}", flush=True)
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for uid, iid, ts in rows:
        by_user[uid].append((ts, iid))
    seqs: dict[int, list[int]] = {}
    for uid, pairs in by_user.items():
        pairs.sort()
        seen: set[int] = set()
        seq: list[int] = []
        for ts, iid in pairs:
            if iid not in seen:
                seen.add(iid)
                seq.append(iid)
        seqs[uid] = seq
    num_users = len(seqs)
    all_items = sorted({iid for seq in seqs.values() for iid in seq})
    id_map = {orig: idx + 1 for idx, orig in enumerate(all_items)}
    num_items = len(all_items)
    print(f"users={num_users} unique_items={num_items}", flush=True)

    train_samples: list[tuple[list[int], int]] = []
    val_pairs: list[tuple[list[int], int]] = []
    test_pairs: list[tuple[list[int], int]] = []
    for uid, seq in seqs.items():
        mapped = [id_map[i] for i in seq]
        if len(mapped) >= 2:
            test_pairs.append((mapped[:-1], mapped[-1]))
        if len(mapped) >= 3:
            val_pairs.append((mapped[:-2], mapped[-2]))
        prefix = mapped[:-2]
        for t in range(1, len(prefix)):
            train_samples.append((prefix[:t], prefix[t]))

    def pad(seq: list[int]) -> list[int]:
        s = seq[-MAX_SEQ:]
        return s + [0] * (MAX_SEQ - len(s))

    train_x = torch.tensor([pad(s) for s, _ in train_samples], dtype=torch.long)
    train_y = torch.tensor([t for _, t in train_samples], dtype=torch.long)
    val_x = torch.tensor([pad(s) for s, _ in val_pairs], dtype=torch.long)
    val_y = torch.tensor([t for _, t in val_pairs], dtype=torch.long)
    test_x = torch.tensor([pad(s) for s, _ in test_pairs], dtype=torch.long)
    test_y = torch.tensor([t for _, t in test_pairs], dtype=torch.long)
    print(f"train={train_x.size(0)} val={val_x.size(0)} test={test_x.size(0)}", flush=True)

    processed = OUT / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    torch.save({"train_x": train_x, "train_y": train_y,
                "val_x": val_x, "val_y": val_y,
                "test_x": test_x, "test_y": test_y}, processed / "datasets.pt")

    texts = load_item_texts()
    item_texts = {str(idx): texts.get(orig, "") for idx, orig in enumerate(all_items, start=1)}
    (processed / "item_texts.json").write_text(json.dumps(item_texts, indent=1), encoding="utf-8")
    id_mapping = {str(idx): orig for idx, orig in enumerate(all_items, start=1)}
    (processed / "id_mapping.json").write_text(json.dumps(id_mapping, indent=1), encoding="utf-8")

    stats = {
        "dataset_name": "MovieLens-1M",
        "protocol_version": "planrec_loo_rightpad_v1",
        "split_protocol": "chronological_leave_one_out",
        "padding_side": "right",
        "max_seq_len": MAX_SEQ,
        "num_users": num_users,
        "num_items": num_items,
        "num_train_samples": train_x.size(0),
        "num_val_samples": val_x.size(0),
        "num_test_samples": test_x.size(0),
        "raw_sha256": "",
        "source_url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    }
    (processed / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1), flush=True)
    print(f"[done] {processed}", flush=True)

if __name__ == "__main__":
    main()
