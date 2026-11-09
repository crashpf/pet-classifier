"""Train a ResNet18 classifier on Oxford-IIIT Pet.

Each epoch trains on the train split, then measures the model on the
validation split. The epoch with the best validation accuracy is saved
to disk, so the saved model is the best one this run produced, not
necessarily the last one.

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
    """Seed every random number generator we use, so runs are repeatable.

    This covers weight init of the new fc layer, data shuffling and augmentation.
    """
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
    """One pass over the training data. Updates the model's weights in place.

    Returns:
        {"loss": mean loss per sample, "acc": accuracy in [0, 1]}
    """
    # train()/eval() don't decide WHAT learns (requires_grad does that); they only
    # switch how BatchNorm and Dropout behave. With a frozen backbone we use eval
    # mode so BatchNorm keeps its pretrained ImageNet statistics instead of
    # drifting every batch. The fc head is a plain Linear layer, so it trains
    # exactly the same in either mode.
    if freeze_backbone:
        model.eval()
    else:
        model.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        # core loop
        optimizer.zero_grad()  # gradients accumulate by default, so reset them
        outputs = model(inputs)  # [batch, num_classes]: one score per breed
        loss = loss_fn(outputs, targets)  # average loss over the batch
        loss.backward()  # compute gradients for every trainable parameter
        optimizer.step()  # update the weights to reduce the loss

        # The loss above is a batch average. Turn it back into a batch total so
        # that, when we divide by total_samples at the end, every image counts
        # equally (the last batch is usually smaller than the rest).
        # .item() converts to a plain float so we don't keep the graph in memory.
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        # argmax over the class scores = predicted breed for each image
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


@torch.inference_mode()  # no gradients tracked: less memory and faster
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Measure the model on a held-out loader (val or test). No weights change.

    Same bookkeeping as train_epoch, minus the zero_grad/backward/step updates.

    Returns:
        {"loss": mean loss per sample, "acc": accuracy in [0, 1]}
    """
    model.eval()  # BatchNorm uses its stored statistics, Dropout is off
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
    """Save the weights plus everything inference needs to rebuild the model.

    Overwrites any existing file at `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # How to rebuild the model: architecture + output size, then load weights.
            "arch": "resnet18",
            "num_classes": len(classes),
            "model_state_dict": model.state_dict(),
            # Maps the model's output index to a breed name, e.g. 12 -> "beagle".
            "classes": classes,
            # Info about how this checkpoint was produced.
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
    """Full training run. Saves the best epoch (by val accuracy) to `output_path`.

    These arguments are what will move into the config file later.

    Returns:
        The best epoch's validation metrics: {"epoch", "loss", "acc"}.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    set_seed(seed)
    device = get_device()
    output_path = Path(output_path)
    logger.info("Using device: %s", device)

    # The test loader is ignored here on purpose since it's only used once, in the final evaluation.
    train_loader, val_loader, _, classes = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    # Move the model to the device before creating the optimizer, so the
    # optimizer holds the parameters that actually live on that device.
    model = get_model(num_classes=len(classes), freeze_backbone=freeze_backbone).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Only pass parameters that can learn. With a frozen backbone that's just
    # the fc head (~19k of ~11M parameters).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in trainable_params):,}")

    # Best validation result so far. -1.0 is lower than any real accuracy 
    # so epoch 1 is guaranteed to become the first "best".
    best: dict[str, float] = {"acc": -1.0}

    # Full loop: train, validate, then keep the model if it's the best yet
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, freeze_backbone
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Train metrics are only logged. Comparing them with val metrics by eye
        # shows overfitting: train keeps improving while val stalls or gets worse.
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f | %.0fs",
            epoch, epochs,
            train_metrics["loss"], train_metrics["acc"],
            val_metrics["loss"], val_metrics["acc"],
            time.perf_counter() - start,
        )

        # Decide using validation accuracy only. If this epoch beats every earlier one,
        # overwrite the checkpoint. The model in memory keeps training afterwards,
        # so after the loop the best weights exist only in the saved file.
        if val_metrics["acc"] > best["acc"]:
            best = {"epoch": epoch, **val_metrics}
            save_checkpoint(output_path, model, classes, epoch, val_metrics, freeze_backbone)
            logger.info("New best val acc %.3f, saved to %s", best["acc"], output_path)

    logger.info("Best val acc %.3f at epoch %d", best["acc"], best["epoch"])
    return best


def main() -> None:
    # Configure logging here (the entry point), not at import time, so importing
    # this module from elsewhere doesn't change that program's log output.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    train()


if __name__ == "__main__":
    main()
