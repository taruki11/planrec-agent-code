import math
import torch


def ranking_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: int = 10,
) -> dict[str, float]:
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected logits [batch, items] and targets [batch]")
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    matches = topk.eq(targets.unsqueeze(1))
    hits = matches.any(dim=1)
    ranks = matches.float().argmax(dim=1) + 1
    ranks = torch.where(hits, ranks, torch.zeros_like(ranks))

    reciprocal = torch.where(hits, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float))
    ndcg = torch.where(
        hits,
        1.0 / torch.log2(ranks.float() + 1.0),
        torch.zeros_like(ranks, dtype=torch.float),
    )
    return {
        f"hit_rate@{k}": hits.float().mean().item(),
        f"mrr@{k}": reciprocal.mean().item(),
        f"ndcg@{k}": ndcg.mean().item(),
    }


def compute_prefix_survival_rate(
    predicted_codes: torch.Tensor,
    target_codes: torch.Tensor,
    level: int = 1,
) -> float:
    """Compute whether the target code prefix at specific level survives in predictions."""
    pred_prefix = predicted_codes[:, :level]
    target_prefix = target_codes[:, :level]
    survived = pred_prefix.eq(target_prefix).all(dim=1)
    return float(survived.float().mean().item())


def compute_codebook_utilization(
    assigned_codes: torch.Tensor,
    num_clusters: int,
    level: int = 0,
) -> float:
    """Compute percentage of active codebook clusters utilized."""
    unique_active = torch.unique(assigned_codes[:, level]).numel()
    return float(unique_active / num_clusters)
