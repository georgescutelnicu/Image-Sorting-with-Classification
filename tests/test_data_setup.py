import os
from torchvision import transforms


def test_create_dataloaders(dataloaders):
    train_loader, test_loader, class_names = dataloaders

    assert len(train_loader) > 0
    assert len(test_loader) > 0
    assert class_names == ["rain", "snow"]

    for images, _ in train_loader:
        assert images.shape[1:] == (3, 224, 224)

    for images, _ in test_loader:
        assert images.shape[1:] == (3, 224, 224)
