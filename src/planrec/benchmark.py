import copy
import time
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Calculate total trainable parameters of a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 10,
    num_runs: int = 100,
) -> float:
    """Measure average single-request inference latency in milliseconds."""
    model.eval()
    sample_input = sample_input.to(device)
    if sample_input.ndim == 1:
        sample_input = sample_input.unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timing runs
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    return (total_time / num_runs) * 1000.0  # ms per request


def measure_inference_latency_sync(
    forward_fn,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 20,
    num_runs: int = 200,
) -> dict[str, float]:
    """Per-request synchronized latency distribution (PB-Beauty-v2 R3).

    Every timed forward is bracketed by ``torch.cuda.synchronize()`` on CUDA,
    so the measured deltas are valid GPU-latency deltas, not asynchronous host
    timings.  Returns ``mean_ms``, ``p50_ms``, ``p95_ms`` plus protocol
    metadata.  ``forward_fn`` is a callable taking the (batched) input; use a
    closure to bypass default-module-``forward`` when the model needs extra
    args (e.g. planner depth).
    """
    sample_input = sample_input.to(device)
    if sample_input.ndim == 1:
        sample_input = sample_input.unsqueeze(0)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = forward_fn(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        trials: list[float] = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            _ = forward_fn(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            trials.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    trials.sort()
    p50 = trials[len(trials) // 2]
    p95 = trials[min(int(len(trials) * 0.95), len(trials) - 1)]
    mean = sum(trials) / len(trials)
    return {
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "warmup_runs": warmup_runs,
        "num_runs": num_runs,
    }


class EarlyStopping:
    """Monitor validation performance and save best checkpoint."""

    def __init__(self, patience: int = 5, delta: float = 1e-5) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score: float | None = None
        self.early_stop: bool = False
        self.counter: int = 0
        self.best_state: dict[str, Any] | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return True

        if score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def load_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
