"""Tests for pet_classifier.dataset.

Run with:
    uv run pytest
"""

import pytest

from pet_classifier.dataset import _split_indices, get_dataloaders

# Fast tests: no images involved, just the splitting logic.
def test_split_has_no_overlap_and_covers_everything():
    train_idx, val_idx = _split_indices(n=100, val_fraction=0.2, seed=42)

    assert len(val_idx) == 20
    assert len(train_idx) == 80
    assert set(train_idx).isdisjoint(val_idx)  # no image in both sets
    assert set(train_idx) | set(val_idx) == set(range(100))  # every image used


def test_split_is_reproducible():
    # Same seed -> same split. This is what makes runs comparable.
    assert _split_indices(100, 0.2, seed=42) == _split_indices(100, 0.2, seed=42)
    # Different seed -> different split.
    assert _split_indices(100, 0.2, seed=42) != _split_indices(100, 0.2, seed=7)


# Slower tests: use the real dataset in data/.

@pytest.fixture(scope="module")
def dataloaders():
    """Build the dataloaders ONCE and share them with every test below.

    num_workers=0 loads in the main process, which starts faster for a test.
    """
    return get_dataloaders(batch_size=8, num_workers=0)


def test_split_sizes(dataloaders):
    train_loader, val_loader, test_loader, classes = dataloaders

    assert len(classes) == 37
    assert len(train_loader.dataset) + len(val_loader.dataset) == 3680  # whole trainval split
    assert len(test_loader.dataset) == 3669 # test split


def test_batch_shapes(dataloaders):
    train_loader, val_loader, test_loader, _ = dataloaders

    for dataloader in (train_loader, val_loader, test_loader):
        images, labels = next(iter(dataloader))
        assert images.shape == (8, 3, 224, 224)  # batch, RGB, height, width
        assert labels.shape == (8,)
        assert labels.min() >= 0 and labels.max() < 37  # valid class indices
