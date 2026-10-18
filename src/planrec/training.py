import torch
from torch import nn
from torch.utils.data import DataLoader

from planrec.models.sasrec import SASRec


def train_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    for item_sequence, target in loader:
        item_sequence = item_sequence.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(item_sequence), target)
        loss.backward()
        optimizer.step()
        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples

