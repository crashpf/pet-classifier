"""Train a ResNet18 classifier on Oxford-IIIT Pet.

How to run:
    uv run python -m pet_classifier.train
"""

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_classifier.dataset import get_dataloaders
from pet_classifier.model import get_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freeze_backbone: bool,
) -> dict[str, float]:
    """One pass over the training data. Updates the model in place.

    Returns:
        {"loss": mean loss per sample, "acc": accuracy in [0, 1]}
    """
    # With a frozen backbone, keep the whole model in eval mode: BatchNorm would
    # otherwise keep updating its running mean/var (not a gradient update, so
    # requires_grad=False doesn't stop it). The new fc layer has no
    # BatchNorm/Dropout, so eval mode doesn't change how it trains.
    if freeze_backbone:
        model.eval()
    else:
        model.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    # Iterate over batches of data in the validation loader
    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        # Core steps
        optimizer.zero_grad() # clear gradients from previous batch
        outputs = model(inputs) # forward pass
        loss = loss_fn(outputs, targets) # cross entropy loss to measure how wrong the prediction is
        loss.backward() # compute gradients that are stored in param.grad
        optimizer.step() # update weights

        # loss is the MEAN over the batch; multiply back by batch size so the
        # smaller last batch doesn't count as much as a full one.
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


@torch.inference_mode()  # no autograd graph: less memory, faster
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """One pass over a held-out loader (val or test) with no weight updates.

    Returns:
        {"loss": mean loss per sample, "acc": accuracy in [0, 1]}
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="eval", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    classes: list[str],
    epoch: int,
    val_metrics: dict[str, float],
    freeze_backbone: bool,
) -> None:
    """Save weights plus everything inference needs to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "resnet18",
            "model_state_dict": model.state_dict(),
            "classes": classes,  # index -> breed name, needed by inference/API
            "num_classes": len(classes),
            "freeze_backbone": freeze_backbone,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def train(
    data_dir: str = "data",
    output_path: str = "models/best.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    num_workers: int = 2,
    seed: int = 42,
) -> dict[str, float]:
    """Full training run. Saves the best model (by val accuracy) to `output_path`.

    These arguments are exactly what will move into the config file later.

    Returns:
        The best epoch's validation metrics: {"epoch", "loss", "acc"}.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    set_seed(seed)
    device = get_device()
    output_path = Path(output_path)
    logger.info("Using device: %s", device)

    train_loader, val_loader, _, classes = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    # Move to device BEFORE building the optimizer.
    model = get_model(num_classes=len(classes), freeze_backbone=freeze_backbone).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Only give the optimizer parameters that can actually learn
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in trainable_params):,}")

    best: dict[str, float] = {"acc": -1.0}

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, freeze_backbone
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f | %.0fs",
            epoch, epochs,
            train_metrics["loss"], train_metrics["acc"],
            val_metrics["loss"], val_metrics["acc"],
            time.perf_counter() - start,
        )

        if val_metrics["acc"] > best["acc"]:
            best = {"epoch": epoch, **val_metrics}
            save_checkpoint(output_path, model, classes, epoch, val_metrics, freeze_backbone)
            logger.info("New best val acc %.3f, saved to %s", best["acc"], output_path)

    logger.info("Best val acc %.3f at epoch %d", best["acc"], best["epoch"])
    return best


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    train()


if __name__ == "__main__":
    main()
