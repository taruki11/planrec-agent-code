from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlanSchema:
    depth: int  # 0: Direct, 1: Plan-1, 2: Plan-2
    num_intent_clusters: int = 4
    num_interest_clusters: int = 4


def validate_plan_structure(plan_tokens: list[int], depth: int) -> bool:
    """Validate whether a predicted plan token sequence adheres to the defined depth schema."""
    if depth == 0:
        return len(plan_tokens) == 0
    elif depth == 1:
        return len(plan_tokens) == 1 and plan_tokens[0] > 0
    elif depth == 2:
        return len(plan_tokens) == 2 and all(t > 0 for t in plan_tokens)
    return False


def compute_plan_item_consistency(
    predicted_plans: list[tuple[int, ...]],
    target_items: list[int],
    item_to_code: dict[int, tuple[int, int]],
) -> float:
    """Compute consistency ratio between predicted plan intents and target item codes."""
    if not predicted_plans or not target_items:
        return 0.0

    matches = 0
    for plan, item_id in zip(predicted_plans, target_items):
        if item_id in item_to_code:
            c1, _ = item_to_code[item_id]
            # Consistency: the planner emits the tokenizer's actual c1 token;
            # do not apply a hard-coded modulo mapping.
            if plan and plan[0] == c1:
                matches += 1

    return matches / len(predicted_plans)
