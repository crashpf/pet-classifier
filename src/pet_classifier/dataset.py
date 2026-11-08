"""Transformations and DataLoaders for the Oxford-IIIT Pet dataset.

How to run:
    uv run python -m pet_classifier.dataset
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


# Transformations function
def define_transformations(
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    image_size: int = 224,
    resize_size: int = 256,
):
    
    train_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Same resize as training, then a deterministic center crop, so objects
    # appear at a similar scale at train and test time.
    test_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, test_transforms


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffles [0, n-1] with a fixed seed and cuts it into train/val index lists.

    The dedicated generator means the split is identical on every run.
    """
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 2,
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """Return (train_loader, val_loader, test_loader, classes)."""
    train_transforms, test_transforms = define_transformations()

    # trainval_aug & trainval_plain contain the same images in the same order
    trainval_aug = OxfordIIITPet(
        root=data_dir,
        split="trainval", 
        target_types="category",
        transform=train_transforms, 
        download=True,
    )
    trainval_plain = OxfordIIITPet(
        root=data_dir, 
        split="trainval", 
        target_types="category",
        transform=test_transforms, 
        download=True,
    )
    test_dataset = OxfordIIITPet(
        root=data_dir, 
        split="test", 
        target_types="category",
        transform=test_transforms, 
        download=True,
    )
    # Shuffle index numbers and split the list
    train_idx, val_idx = _split_indices(len(trainval_aug), val_fraction, seed)
    # train_dataset maps train_idx into trainval_aug so the training images get augmented
    train_dataset = Subset(trainval_aug, train_idx)
    # val_dataset maps val_idx into trainval_plain so the validation images stay as is
    val_dataset = Subset(trainval_plain, val_idx)

    # Create the dataloaders with the new datasets
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = trainval_aug.classes
    return train_dataloader, val_dataloader, test_dataloader, classes


def main():
    """Smoke test: check split sizes and that one batch from each split loads."""
    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders()

    print(f"Classes: {len(classes)}")
    print(f"Train images: {len(train_dataloader.dataset)}")
    print(f"Val images:   {len(val_dataloader.dataset)}")
    print(f"Test images:  {len(test_dataloader.dataset)}")

    # Pull one batch from each split to confirm shapes and that
    # both the augmented and plain transforms work.
    train_images, train_labels = next(iter(train_dataloader))
    val_images, val_labels = next(iter(val_dataloader))
    test_images, test_labels = next(iter(test_dataloader))

    print(f"Train batch: images {train_images.shape}, labels {train_labels.shape}")
    print(f"Val batch:   images {val_images.shape}, labels {val_labels.shape}")
    print(f"Test batch:  images {test_images.shape}, labels {test_labels.shape}")


if __name__ == "__main__":
    main()
