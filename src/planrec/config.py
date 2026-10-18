from dataclasses import dataclass


@dataclass(frozen=True)
class SASRecConfig:
    num_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_items < 2:
            raise ValueError("num_items must be at least 2")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class GRU4RecConfig:
    num_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_items < 2:
            raise ValueError("num_items must be at least 2")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

