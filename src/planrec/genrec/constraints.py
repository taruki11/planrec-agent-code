from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence


@dataclass
class _Node:
    children: dict[Hashable, "_Node"] = field(default_factory=dict)
    item_id: int | None = None


class TokenSequenceTrie:
    """Prefix constraint over complete item-token sequences."""

    def __init__(self) -> None:
        self.root = _Node()

    def insert(self, sequence: Sequence[Hashable], item_id: int) -> None:
        if not sequence:
            raise ValueError("sequence must not be empty")
        node = self.root
        for token in sequence:
            node = node.children.setdefault(token, _Node())
        if node.item_id is not None and node.item_id != item_id:
            raise ValueError("two items cannot share the same terminal sequence")
        node.item_id = int(item_id)

    def allowed_next(self, prefix: Sequence[Hashable]) -> list[Hashable]:
        node = self.root
        for token in prefix:
            node = node.children.get(token)
            if node is None:
                return []
        return list(node.children)

    def decode(self, sequence: Sequence[Hashable]) -> int | None:
        node = self.root
        for token in sequence:
            node = node.children.get(token)
            if node is None:
                return None
        return node.item_id

    @classmethod
    def from_sequences(cls, sequences: Iterable[tuple[Sequence[Hashable], int]]) -> "TokenSequenceTrie":
        trie = cls()
        for sequence, item_id in sequences:
            trie.insert(sequence, item_id)
        return trie
