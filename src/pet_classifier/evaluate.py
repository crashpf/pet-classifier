"""Evaluate the best saved model on the held-out test set.

Run this ONCE on the model you picked using validation results. If you tweak
things based on the test score and re-run, the test set stops being an honest
measure of performance on unseen pets.

Produces:
    - overall test accuracy (logged)
    - per-breed precision / recall / F1 (logged + reports/classification_report.txt)
    - a confusion matrix image (reports/confusion_matrix.png)

How to run (after training has created models/best.pt):
    uv run python -m pet_classifier.evaluate
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_classifier.dataset import get_dataloaders
from pet_classifier.model import get_model
from pet_classifier.train import get_device

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Rebuild the model from a checkpoint saved by train.save_checkpoint.

    Returns:
        (model in eval mode on `device`, list of class names)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Rebuild the same architecture with the same output size, then load the
    # trained weights into it. freeze_backbone doesn't matter here: it only
    # affects training, and we never train this model.
    model = get_model(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    logger.info(
        "Loaded %s (epoch %d, val acc %.3f)",
        checkpoint_path, checkpoint["epoch"], checkpoint["val_metrics"]["acc"],
    )
    return model, checkpoint["classes"]


@torch.inference_mode()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a whole loader.

    Returns:
        (true labels, predicted labels) as 1-D numpy arrays, one entry per image.
    """
    model.eval()
    all_targets, all_preds = [], []

    for inputs, targets in tqdm(loader, desc="test", leave=False):
        inputs.to(device)
        outputs = model(inputs)  # [batch, num_classes]
        all_preds.append(outputs.argmax(dim=1).cpu())  # predicted class per image
        all_targets.append(targets)

    # Glue the per-batch tensors into one long array each.
    return torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy()


def save_confusion_matrix(cm: np.ndarray, classes: list[str], path: Path) -> None:
    """Save the confusion matrix as an image (rows = true breed, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, include_values=False, cmap="Blues", xticks_rotation=90, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free the memory; we never display it


def run_evaluation(
    checkpoint_path: str = "models/best.pt",
    data_dir: str = "data",
    output_dir: str = "reports",
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, float]:
    """Evaluate a checkpoint on the test set and write the reports.

    Returns:
        {"test_acc": accuracy in [0, 1]}
    """
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first: "
            "uv run python -m pet_classifier.train"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, classes = load_model(checkpoint_path, device)

    # Only the test loader is needed; train/val were used during training.
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )

    y_true, y_pred = predict_all(model, test_loader, device)

    # 1. Overall accuracy: fraction of test images predicted correctly.
    test_acc = float((y_true == y_pred).mean())
    logger.info("Test accuracy: %.3f (%d images)", test_acc, len(y_true))

    # 2. Per-breed metrics. Overall accuracy can hide breeds the model is bad at.
    #    precision: when it says "beagle", how often is it right?
    #    recall: of all real beagles, how many did it find?
    #    f1: a single score that balances the two
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    logger.info("Per-breed report:\n%s", report)
    (output_dir / "classification_report.txt").write_text(report)

    # 3. Confusion matrix: which breeds get mistaken for which.
    #    cm[i, j] = how many images of true breed i were predicted as breed j.
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")
    logger.info("Reports saved to %s/", output_dir)

    return {"test_acc": test_acc}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    run_evaluation()


if __name__ == "__main__":
    main()
