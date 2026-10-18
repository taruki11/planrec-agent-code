from __future__ import annotations

from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


def common_semantic_prefix(left: tuple[int, ...], right: tuple[int, ...], num_levels: int) -> int:
    shared = 0
    for level in range(num_levels):
        if left[level] != right[level]:
            break
        shared += 1
    return shared


def recommendation_reward(
    completion: str,
    target_item: int,
    semantic_ids: ResidualKMeansSemanticTokenizer,
    exact_reward: float = 2.0,
    valid_reward: float = 0.2,
    prefix_reward: float = 0.8,
    invalid_penalty: float = -1.0,
) -> float:
    """Hybrid reward for GRPO-style recommendation post-training.

    Exact next-item prediction receives the largest reward.  Other valid items
    receive only a bounded semantic-prefix shaping reward; invalid semantic IDs
    are penalized.  This avoids treating format validity as recommendation
    success while still giving sparse-policy optimization a useful gradient.
    """

    predicted_item = semantic_ids.item_from_text(completion)
    if predicted_item is None:
        return float(invalid_penalty)
    if predicted_item == int(target_item):
        return float(exact_reward)
    predicted_code = semantic_ids.item_to_code[predicted_item]
    target_code = semantic_ids.item_to_code[int(target_item)]
    shared = common_semantic_prefix(predicted_code, target_code, semantic_ids.num_levels)
    return float(valid_reward + prefix_reward * shared / semantic_ids.num_levels)
