import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ConstrainedDecoderTrie:
    """Trie structure for prefix-constrained autoregressive decoding of item codes."""

    def __init__(self) -> None:
        self.trie: dict[int, set[int]] = defaultdict(set)
        self.valid_codes: set[tuple[int, int]] = set()

    def insert(self, c1: int, c2: int) -> None:
        self.trie[c1].add(c2)
        self.valid_codes.add((c1, c2))

    def get_valid_first_tokens(self) -> list[int]:
        return sorted(list(self.trie.keys()))

    def get_valid_second_tokens(self, c1: int) -> list[int]:
        return sorted(list(self.trie.get(c1, set())))

    def is_valid(self, c1: int, c2: int) -> bool:
        return (c1, c2) in self.valid_codes


class SemanticTokenizer:
    """Hierarchical item tokenizer mapping IDs to unique two-level semantic codes.

    The first code is obtained with k-means over item representations.  Items in
    each first-level cluster are then partitioned by k-means over residuals and
    ordered by local residual radius.  The second token remains
    unique per item (required by the current constrained decoder), while nearby
    second tokens share a residual sub-cluster.  This replaces the previous
    norm-quantile heuristic and makes the codebook deterministic for a fixed
    representation matrix and seed.
    """

    def __init__(self, num_items: int, num_clusters_c1: int = 4, num_clusters_c2: int = 8) -> None:
        self.num_items = num_items
        self.num_clusters_c1 = min(num_clusters_c1, num_items)
        self.num_clusters_c2 = max(1, num_clusters_c2)
        self.item_to_code: dict[int, tuple[int, int]] = {}
        self.code_to_item: dict[tuple[int, int], int] = {}
        self.trie = ConstrainedDecoderTrie()

    @staticmethod
    def _kmeans(
        values: np.ndarray,
        num_clusters: int,
        rng: np.random.Generator,
        max_iter: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small dependency-free k-means with k-means++ initialization."""
        n = values.shape[0]
        k = min(max(1, num_clusters), n)
        if n == 1:
            return np.zeros(1, dtype=np.int64), values.astype(np.float32, copy=True)

        values = values.astype(np.float32, copy=False)
        centers = np.empty((k, values.shape[1]), dtype=np.float32)
        first = int(rng.integers(0, n))
        centers[0] = values[first]
        min_dist = ((values - centers[0]) ** 2).sum(axis=1)
        for idx in range(1, k):
            total = float(min_dist.sum())
            if not np.isfinite(total) or total <= 1e-12:
                next_idx = int(rng.integers(0, n))
            else:
                next_idx = int(rng.choice(n, p=min_dist / total))
            centers[idx] = values[next_idx]
            dist = ((values - centers[idx]) ** 2).sum(axis=1)
            min_dist = np.minimum(min_dist, dist)

        value_norm = (values * values).sum(axis=1, keepdims=True)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            distances = value_norm + (centers * centers).sum(axis=1)[None, :] - 2.0 * values @ centers.T
            new_labels = distances.argmin(axis=1).astype(np.int64)
            new_centers = centers.copy()
            for cluster in range(k):
                members = values[new_labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    # Re-seed empty clusters with the currently worst-covered item.
                    new_centers[cluster] = values[int(distances.min(axis=1).argmax())]
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels, centers = new_labels, new_centers
                break
            labels, centers = new_labels, new_centers
        return labels, centers

    def fit_from_item_embeddings(self, item_embeddings: torch.Tensor, seed: int = 42) -> dict[str, Any]:
        """Fit a 2-level hierarchical quantization mapping using item representations."""
        if item_embeddings.shape[0] < self.num_items + 1:
            raise ValueError("item_embeddings must cover 0..num_items")

        # Exclude padding token 0
        embeddings = item_embeddings[1 : self.num_items + 1].detach().cpu().numpy()
        rng = np.random.default_rng(seed)
        embeddings = np.nan_to_num(embeddings.astype(np.float32, copy=False))
        # Centering prevents a global embedding bias from becoming the only
        # signal used by the first-level partition.
        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

        # Level 1: coarse semantic partition in representation space.
        c1_labels, c1_centers = self._kmeans(embeddings, self.num_clusters_c1, rng)
        c1_assignments = c1_labels + 1

        # Level 2: residual sub-clusters followed by a deterministic local order.
        c2_counters: defaultdict[int, int] = defaultdict(int)
        code_collisions = 0

        for c1_zero in range(self.num_clusters_c1):
            member_indices = np.flatnonzero(c1_labels == c1_zero)
            if len(member_indices) == 0:
                continue
            c1 = c1_zero + 1
            residuals = embeddings[member_indices] - c1_centers[c1_zero]
            sub_k = min(self.num_clusters_c2, len(member_indices))
            sub_labels, sub_centers = self._kmeans(residuals, sub_k, rng)
            for sub in sorted(range(sub_k), key=lambda s: tuple(sub_centers[s].tolist())):
                local = member_indices[sub_labels == sub]
                if len(local) == 0:
                    continue
                local_residuals = residuals[sub_labels == sub]
                if len(local) > 1:
                    # Distance from the residual sub-centroid is a cheap,
                    # deterministic local order and avoids an SVD per group.
                    local_radius = ((local_residuals - sub_centers[sub]) ** 2).sum(axis=1)
                    local = local[np.argsort(local_radius, kind="stable")]
                for item_idx in local:
                    item_id = int(item_idx) + 1
                    c2_counters[c1] += 1
                    c2 = c2_counters[c1]
                    code = (c1, c2)
                    if code in self.code_to_item:
                        code_collisions += 1
                    self.item_to_code[item_id] = code
                    self.code_to_item[code] = item_id
                    self.trie.insert(c1, c2)

        # Statistics
        num_codes_used = len(self.valid_codes)
        max_c2 = max(c2_counters.values(), default=1)
        utilization = num_codes_used / (self.num_clusters_c1 * max_c2)

        return {
            "algorithm": "hierarchical_kmeans_residual_order_v1",
            "num_items": self.num_items,
            "num_clusters_c1": self.num_clusters_c1,
            "num_clusters_c2": self.num_clusters_c2,
            "max_c2_per_c1": max_c2,
            "code_collisions": code_collisions,
            "code_utilization": utilization,
            "num_valid_codes": num_codes_used,
        }

    @property
    def valid_codes(self) -> set[tuple[int, int]]:
        return self.trie.valid_codes

    @property
    def max_c2(self) -> int:
        """Largest valid second-level token, including uneven c1 cluster sizes."""
        return max((code[1] for code in self.valid_codes), default=1)

    def save(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "item_to_code": {str(k): list(v) for k, v in self.item_to_code.items()},
            "num_items": self.num_items,
            "num_clusters_c1": self.num_clusters_c1,
            "num_clusters_c2": self.num_clusters_c2,
            "algorithm": "hierarchical_kmeans_residual_order_v1",
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.num_items = data["num_items"]
        self.num_clusters_c1 = data["num_clusters_c1"]
        self.num_clusters_c2 = data.get("num_clusters_c2", 1)
        self.item_to_code = {int(k): (v[0], v[1]) for k, v in data["item_to_code"].items()}
        self.code_to_item = {v: k for k, v in self.item_to_code.items()}
        self.trie = ConstrainedDecoderTrie()
        for c1, c2 in self.code_to_item.keys():
            self.trie.insert(c1, c2)
