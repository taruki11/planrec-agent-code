import torch

from planrec.uncertainty import normalized_entropy, top2_margin


def extract_router_features(
    model: torch.nn.Module,
    item_sequence: torch.Tensor,
    max_seq_len: int = 50,
) -> torch.Tensor:
    """Extract 5-dimensional uncertainty and sequence features for the Router."""
    model.eval()
    with torch.no_grad():
        raw_logits = model(item_sequence)
        item_logits = raw_logits[:, 1:]

        # 1. Normalized Entropy
        ent = normalized_entropy(item_logits).unsqueeze(1)
        # 2. Top-2 Margin
        mar = top2_margin(item_logits).unsqueeze(1)
        seq_lens_1d = item_sequence.ne(0).sum(dim=1)
        seq_lens = (seq_lens_1d.float() / max_seq_len).unsqueeze(1)
        # 4. Last-item representation norm.  Do not use the raw item ID as a
        # proxy: IDs are arbitrary and would leak the deterministic mapping
        # rather than measuring representation uncertainty.
        last_indices = (seq_lens_1d - 1).clamp_min(0)
        last_items = item_sequence[torch.arange(item_sequence.size(0), device=item_sequence.device), last_indices]
        item_embedding = getattr(model, "item_embedding", None)
        if item_embedding is not None:
            last_vectors = item_embedding(last_items)
            item_norm = last_vectors.norm(dim=-1, keepdim=True) / (last_vectors.size(-1) ** 0.5)
        else:
            item_norm = torch.zeros_like(seq_lens)
        # 5. Logits Max Confidence
        max_conf = item_logits.softmax(dim=-1).max(dim=-1, keepdim=True).values

        features = torch.cat([ent, mar, seq_lens, item_norm, max_conf], dim=1)
        return torch.nan_to_num(features, nan=0.0)
