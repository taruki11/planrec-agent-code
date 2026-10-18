#!/usr/bin/env python3
"""MovieLens-1M专属 Semantic ID：冻结 embeddings -> RKM 3 级 -> 唯一可逆 round-trip。

输出 data/movielens_1m/processed/semantic_ids.json（结构与 Beauty 版一致），
并验证 3706/3706 唯一可逆。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


def main() -> None:
    processed = ROOT / "data" / "movielens_1m" / "processed"
    emb_path = processed / "item_embeddings.pt"
    out = processed / "semantic_ids.json"

    emb = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(emb, dict):
        emb = emb.get("embeddings", emb)
    num_items = int(emb.size(0)) - 1  # row 0 = padding
    print(f"embeddings shape={tuple(emb.shape)} num_items={num_items}", flush=True)

    tok = ResidualKMeansSemanticTokenizer(num_items=num_items, num_levels=3, codebook_size=256)
    stats = tok.fit(emb, seed=42)
    print(f"fit stats: {json.dumps(stats)[:300]}", flush=True)
    tok.save(out)

    # round-trip 唯一性校验
    loaded = ResidualKMeansSemanticTokenizer.load(out)
    reverse: dict[str, int] = {}
    bad = 0
    for item_id in range(1, num_items + 1):
        tokens = loaded.tokens_for_item(item_id)
        dec = loaded.item_from_tokens(tokens)
        if dec != item_id:
            bad += 1
        key = tuple(tokens)
        if key in reverse:
            bad += 1
        reverse[key] = item_id
    print(f"ROUNDTRIP bad={bad}/{num_items}", flush=True)
    assert bad == 0, "SID round-trip failed"
    print(f"[done] {out} (round-trip OK {num_items}/{num_items})", flush=True)


if __name__ == "__main__":
    main()
