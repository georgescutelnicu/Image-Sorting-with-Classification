import os
import torch
import pytest
from torchvision import transforms
from data_setup import create_dataloaders
from model_builder import create_effnetb2_model


@pytest.fixture
def dataloaders():
    batch_size = 4
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir='tests/data/train',
        test_dir='tests/data/test',
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        batch_size=batch_size,
        num_workers=os.cpu_count()
    )
    return train_loader, test_loader, class_names


@pytest.fixture
def model():
    model, transforms = create_effnetb2_model(num_classes=2)
    return model.to("cpu"), transforms


@pytest.fixture
def loss_fn():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model):
    model, _ = model
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def device():
    return "cpu"
