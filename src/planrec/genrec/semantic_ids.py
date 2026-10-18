from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


FORMAT_TOKENS = (
    "<rec_history>",
    "</rec_history>",
    "<rec_target>",
    "</rec_target>",
    "<item>",
    "</item>",
)


class ResidualKMeansSemanticTokenizer:
    """Map item embeddings to unique, fixed-length residual-quantized IDs.

    Each item receives ``num_levels`` semantic codes plus one collision code.
    The collision code is zero for the first item with a semantic code and is
    incremented only when multiple items share the same quantized path.  The
    final sequence is therefore always reversible while early tokens retain
    the coarse-to-fine geometry of residual quantization.
    """

    _SID_PATTERN = re.compile(r"^<sid_l(?P<level>\d+)_(?P<code>\d+)>$")
    _COLLISION_PATTERN = re.compile(r"^<sid_c_(?P<code>\d+)>$")

    def __init__(self, num_items: int, num_levels: int = 3, codebook_size: int = 256) -> None:
        if num_items < 1:
            raise ValueError("num_items must be positive")
        if num_levels < 1:
            raise ValueError("num_levels must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        self.num_items = int(num_items)
        self.num_levels = int(num_levels)
        self.codebook_size = int(codebook_size)
        self.codebooks: list[np.ndarray] = []
        self.item_to_code: dict[int, tuple[int, ...]] = {}
        self.code_to_item: dict[tuple[int, ...], int] = {}
        self.max_collision = 0

    @staticmethod
    def _kmeans(
        values: np.ndarray,
        num_clusters: int,
        rng: np.random.Generator,
        max_iter: int = 40,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = values.shape[0]
        k = min(max(1, num_clusters), n)
        values = values.astype(np.float32, copy=False)
        centers = np.empty((k, values.shape[1]), dtype=np.float32)
        first = int(rng.integers(0, n))
        centers[0] = values[first]
        min_dist = ((values - centers[0]) ** 2).sum(axis=1)
        for index in range(1, k):
            total = float(min_dist.sum())
            candidate = int(rng.integers(0, n)) if total <= 1e-12 else int(rng.choice(n, p=min_dist / total))
            centers[index] = values[candidate]
            min_dist = np.minimum(min_dist, ((values - centers[index]) ** 2).sum(axis=1))

        labels = np.full(n, -1, dtype=np.int64)
        norms = (values * values).sum(axis=1, keepdims=True)
        for _ in range(max_iter):
            distances = norms + (centers * centers).sum(axis=1)[None, :] - 2.0 * values @ centers.T
            new_labels = distances.argmin(axis=1).astype(np.int64)
            new_centers = centers.copy()
            for cluster in range(k):
                members = values[new_labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    new_centers[cluster] = values[int(distances.min(axis=1).argmax())]
            if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
                labels, centers = new_labels, new_centers
                break
            labels, centers = new_labels, new_centers
        return labels, centers

    def fit(self, item_embeddings: torch.Tensor | np.ndarray, seed: int = 42) -> dict[str, float | int | str]:
        values = item_embeddings.detach().cpu().numpy() if isinstance(item_embeddings, torch.Tensor) else np.asarray(item_embeddings)
        if values.ndim != 2 or values.shape[0] < self.num_items + 1:
            raise ValueError("item_embeddings must have shape [num_items + 1, dim]")
        values = np.nan_to_num(values[1 : self.num_items + 1].astype(np.float32, copy=False))
        values -= values.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.clip(scale, 1e-6, None)

        residual = values.copy()
        all_codes = np.empty((self.num_items, self.num_levels), dtype=np.int64)
        self.codebooks = []
        for level in range(self.num_levels):
            labels, centers = self._kmeans(
                residual,
                self.codebook_size,
                np.random.default_rng(seed + level),
            )
            all_codes[:, level] = labels
            self.codebooks.append(centers)
            residual = residual - centers[labels]

        grouped: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_id, raw_code in enumerate(all_codes, start=1):
            grouped[tuple(int(value) for value in raw_code)].append(item_id)

        self.item_to_code.clear()
        self.code_to_item.clear()
        self.max_collision = 0
        collision_items = 0
        for raw_code, item_ids in grouped.items():
            if len(item_ids) > 1:
                collision_items += len(item_ids)
            for collision, item_id in enumerate(sorted(item_ids)):
                code = raw_code + (collision,)
                self.item_to_code[item_id] = code
                self.code_to_item[code] = item_id
                self.max_collision = max(self.max_collision, collision)

        reconstruction_mse = float(np.mean(residual * residual))
        return {
            "algorithm": "residual_kmeans_unique_v1",
            "num_items": self.num_items,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "max_collision": self.max_collision,
            "collision_items": collision_items,
            "collision_rate": collision_items / self.num_items,
            "reconstruction_mse": reconstruction_mse,
        }

    @staticmethod
    def level_token(level: int, code: int) -> str:
        return f"<sid_l{level}_{code:04d}>"

    @staticmethod
    def collision_token(code: int) -> str:
        return f"<sid_c_{code:04d}>"

    def tokens_for_item(self, item_id: int) -> list[str]:
        code = self.item_to_code[int(item_id)]
        semantic = [self.level_token(level, code[level]) for level in range(self.num_levels)]
        return semantic + [self.collision_token(code[-1])]

    def completion_tokens_for_item(self, item_id: int) -> list[str]:
        return ["<item>", *self.tokens_for_item(item_id), "</item>", "</rec_target>"]

    def item_from_tokens(self, tokens: list[str]) -> int | None:
        semantic_codes: dict[int, int] = {}
        collision: int | None = None
        for token in tokens:
            level_match = self._SID_PATTERN.match(token)
            if level_match:
                semantic_codes[int(level_match.group("level"))] = int(level_match.group("code"))
                continue
            collision_match = self._COLLISION_PATTERN.match(token)
            if collision_match:
                collision = int(collision_match.group("code"))
        if collision is None or set(semantic_codes) != set(range(self.num_levels)):
            return None
        code = tuple(semantic_codes[level] for level in range(self.num_levels)) + (collision,)
        return self.code_to_item.get(code)

    def item_from_text(self, text: str) -> int | None:
        return self.item_from_tokens(text.replace("\n", " ").split())

    def added_tokens(self) -> list[str]:
        tokens = list(FORMAT_TOKENS)
        for level, codebook in enumerate(self.codebooks):
            tokens.extend(self.level_token(level, code) for code in range(len(codebook)))
        tokens.extend(self.collision_token(code) for code in range(self.max_collision + 1))
        return tokens

    def save(self, path: str | Path) -> None:
        if len(self.item_to_code) != self.num_items:
            raise RuntimeError("fit must be called before save")
        payload = {
            "algorithm": "residual_kmeans_unique_v1",
            "num_items": self.num_items,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "max_collision": self.max_collision,
            "item_to_code": {str(item): list(code) for item, code in self.item_to_code.items()},
            "codebooks": [codebook.tolist() for codebook in self.codebooks],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ResidualKMeansSemanticTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["num_items"], payload["num_levels"], payload["codebook_size"])
        tokenizer.max_collision = int(payload["max_collision"])
        tokenizer.item_to_code = {
            int(item): tuple(int(value) for value in code)
            for item, code in payload["item_to_code"].items()
        }
        tokenizer.code_to_item = {code: item for item, code in tokenizer.item_to_code.items()}
        tokenizer.codebooks = [np.asarray(codebook, dtype=np.float32) for codebook in payload["codebooks"]]
        return tokenizer
