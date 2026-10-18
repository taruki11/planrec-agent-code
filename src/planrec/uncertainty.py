import math

import torch


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(-1) < 2:
        raise ValueError("at least two outcomes are required")
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    # Mathematically lim_{p->0} p*log(p) = 0. Avoid 0.0 * (-inf) = NaN
    safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=-1)
    return entropy / math.log(logits.size(-1))


def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(-1) < 2:
        raise ValueError("at least two outcomes are required")
    top2 = logits.softmax(dim=-1).topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]
