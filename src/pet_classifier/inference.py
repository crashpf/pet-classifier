"""Predict the breed of a single pet image.

How to run:
    uv run python -m pet_classifier.inference path/to/image.jpg
"""

import sys

import torch
from PIL import Image
from torch import nn

from pet_classifier.dataset import define_transformations
from pet_classifier.model import load_model
from pet_classifier.train import get_device


@torch.inference_mode()  # no gradients needed: we only predict
def predict(
    model: nn.Module,
    classes: list[str],
    image: Image.Image,
    transform,
    device: torch.device,
    top_k: int = 3,
) -> list[dict]:
    """Return the top_k most likely breeds for one image.

    Takes an already-loaded model and an already-opened image, so it can be
    called many times (e.g. once per API request) without reloading anything.

    Returns:
        [{"breed": "Beagle", "probability": 0.93}, ...] most likely first.
    """
    # Model needs exactly 3 colour channels; PNGs may have 4, grayscale photos 1.
    image = image.convert("RGB")

    # Same preprocessing as val/test, then add a batch dimension:
    # [3, 224, 224] -> [1, 3, 224, 224]
    batch = transform(image).unsqueeze(0).to(device)

    logits = model(batch)  # [1, num_classes] raw scores

    # dim=1 is the class dimension, so each image's scores become probabilities
    # that sum to 1. (dim=0 would normalize over the batch of 1 -> all 100%.)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)  # each [1, top_k]

    # [0] selects our single image; .item() gives plain Python numbers (JSON-friendly).
    return [
        {"breed": classes[idx.item()], "probability": prob.item()}
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]


def main() -> None:
    # Read the image path from the command line: argv[0] is the script, argv[1] the image.
    if len(sys.argv) != 2:
        print("Usage: uv run python -m pet_classifier.inference path/to/image.jpg")
        sys.exit(1)
    image_path = sys.argv[1]

    # Slow setup: done ONCE. A web API would do this at startup.
    device = get_device()
    model, classes = load_model("models/best.pt", device)
    _, test_transform = define_transformations()  # reuse, never copy, the transforms

    image = Image.open(image_path)
    for result in predict(model, classes, image, test_transform, device):
        print(f"{result['breed']:<30} {result['probability']:.1%}")


if __name__ == "__main__":
    main()
