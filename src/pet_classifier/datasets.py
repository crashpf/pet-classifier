from pathlib import Path
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def get_dataloaders(data_dir: str = "data", batch_size: int = 32, num_workers: int = 2):

    # Train Validation set
    train_dataset = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=transform,
        download=True
    )

    # Test set
    test_dataset = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        transform=transform,
        download=True
    )

    # DataLoaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    classes = train_dataset.classes
    return train_dataloader, test_dataloader, classes