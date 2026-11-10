import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_model(num_classes: int = 37, freeze_backbone: bool = True) -> nn.Module:
    """
    This function loads a resNet18 model with default weights and replaces
    its final fully connected layer to match the number of classes in our dataset.
    Args:
        - num_classes (int): Number of output classes in the dataset.
        - freeze_backbone (bool): If True it freezes the pretrained backbone 
            parameters so that only the new fully connected layer is trained.
            Switch to False if you want to fine-tune every model parameter.
            Defaults to True.

    Returns:
        nn.Module: A resNet18 model with pretrained weights and a modified
            fully connected layer for the target classification task. Keep in
            mind that the new fully connected layer needs to be trained.
    """
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be an integer greater than or equal 1, got {num_classes}")
    
    weights = ResNet18_Weights.DEFAULT 
    model = resnet18(weights=weights)

    if freeze_backbone: # If true then freeze the whole model
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # param.requires_grad = True by default when we create a new nn.Module
    return model

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

def main():
    print()
    model = get_model(num_classes=37, freeze_backbone=True) # defaults
    total_params = sum(p.numel() for p in model.parameters()) # total parameters of the model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # trainable parameters of the model
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Smoke test with dummy data to inspect shapes
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()