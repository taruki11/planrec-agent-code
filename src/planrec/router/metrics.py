import numpy as np
import torch
import torch.nn.functional as F


def compute_expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class classification."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute multi-class Brier score."""
    num_classes = probs.size(1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item())


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Negative Log-Likelihood."""
    nll = F.nll_loss(probs.clamp_min(1e-12).log(), labels)
    return float(nll.item())


def positive_precision_recall(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Precision/recall for the positive class (label==1) at a decision threshold.

    PB-Beauty-v2 R2: router selection must use positive-class metrics, because
    accuracy-only selection rewards an all-fast classifier under imbalance.
    An all-negative prediction yields precision 0.0 (reported, not NaN).
    """
    pred_pos = scores >= threshold
    pos = labels == 1
    tp = int((pred_pos & pos).sum().item())
    fp = int((pred_pos & ~pos).sum().item())
    fn = int((~pred_pos & pos).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "tp": tp, "fp": fp, "fn": fn}


def precision_recall_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """PR-AUC over the positive class via the trapezoid rule on ranked scores."""
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    order = torch.argsort(scores, descending=True)
    s_sorted = scores[order]
    y_sorted = (labels == 1).long()[order]

    n_pos = int(y_sorted.sum().item())
    n_neg = int((y_sorted == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    precision_curve = []
    recall_curve = []
    tp = fp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision_curve.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recall_curve.append(tp / n_pos)
    area = 0.0
    for i in range(1, len(precision_curve)):
        dr = recall_curve[i] - recall_curve[i - 1]
        if dr != 0:
            area += dr * (precision_curve[i] + precision_curve[i - 1]) / 2.0
    return float(area)
