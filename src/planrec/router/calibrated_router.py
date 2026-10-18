import torch
from torch import nn
import torch.nn.functional as F


class CalibratedRouter(nn.Module):
    """MLP Router mapping uncertainty features to routing depths with Temperature Scaling."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_depths: int = 3) -> None:
        super().__init__()
        self.num_depths = num_depths
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_depths),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(features)
        return F.softmax(logits / self.temperature, dim=-1)

    def fit_temperature(self, val_features: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Find optimal temperature T on validation set to minimize ECE / NLL."""
        self.eval()
        with torch.no_grad():
            raw_logits = self.forward_logits(val_features)

        best_t = 1.0
        best_nll = float("inf")

        for t in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]:
            probs = F.softmax(raw_logits / t, dim=-1)
            nll = F.nll_loss(probs.clamp_min(1e-12).log(), val_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_t = t

        self.temperature.copy_(torch.tensor(best_t))
        return best_t


class EntropyThresholdRouter:
    """Baseline Router choosing depth via simple normalized entropy thresholds."""

    def __init__(self, t1: float = 0.5, t2: float = 0.8) -> None:
        self.t1 = t1
        self.t2 = t2

    def predict_depth(self, entropy: float) -> int:
        if entropy < self.t1:
            return 0  # High confidence -> Direct
        elif entropy < self.t2:
            return 1  # Medium confidence -> Plan-1
        else:
            return 2  # High uncertainty -> Plan-2
