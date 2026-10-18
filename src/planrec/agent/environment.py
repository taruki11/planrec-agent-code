from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentAction(Enum):
    RETRIEVE_METADATA = 0
    INVOKE_VERIFIER = 1
    FALLBACK_DIRECT = 2
    HALT_AND_OUTPUT = 3


@dataclass
class AgentState:
    step_count: int = 0
    max_budget_steps: int = 3
    remaining_budget: float = 1.0
    current_uncertainty: float = 0.5
    verifier_passed: bool | None = None
    retrieved_metadata: bool = False
    is_halted: bool = False
    chosen_action_history: list[AgentAction] = None

    def __post_init__(self) -> None:
        if self.chosen_action_history is None:
            self.chosen_action_history = []
