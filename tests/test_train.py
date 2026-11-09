"""Tests for pet_classifier.train, using a tiny fake model and random data."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pet_classifier.train import evaluate, train, train_epoch


@pytest.fixture
def tiny_setup():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)  # 4 input features -> 3 classes
    data = TensorDataset(torch.randn(20, 4), torch.randint(0, 3, (20,)))
    loader = DataLoader(data, batch_size=8)  # 20 samples -> batches of 8, 8, 4
    return model, loader, nn.CrossEntropyLoss(), torch.device("cpu")


def test_eval_does_not_change_weights(tiny_setup):
    model, loader, loss_fn, device = tiny_setup
    before = model.weight.clone()

    metrics = evaluate(model, loader, loss_fn, device)

    assert torch.equal(model.weight, before)
    assert set(metrics) == {"loss", "acc"}
    assert 0.0 <= metrics["acc"] <= 1.0


def test_train_changes_weights(tiny_setup):
    model, loader, loss_fn, device = tiny_setup
    before = model.weight.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_epoch(model, loader, loss_fn, optimizer, device, freeze_backbone=False)

    assert not torch.equal(model.weight, before)


def test_train_rejects_zero_epochs():
    with pytest.raises(ValueError):
        train(epochs=0)
