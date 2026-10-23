from pathlib import Path
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms

# Transformations function
def define_transformations(
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
    ):

    train_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transforms, test_transforms

# DataLoaders function
def get_dataloaders(data_dir: str = "data", batch_size: int = 32, num_workers: int = 2):

    train_transforms, test_transforms = define_transformations()
    # Train Validation set
    train_dataset = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=train_transforms,
        download=True
    )

    # Test set
    test_dataset = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        transform=test_transforms,
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

def main():
    print("Running data loading pipeline...")

    train_dataloader, test_dataloader, classes = get_dataloaders(data_dir="data", batch_size=32, num_workers=2)

    image, label = next(iter(train_dataloader))
    print(f"Total classes found: {len(classes)}")
    print(f"Image batch shape: {image.shape}")
    print(f"Label batch shape: {label.shape}")
