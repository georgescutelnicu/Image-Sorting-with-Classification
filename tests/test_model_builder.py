import torch
import torchvision


def test_create_effnetb2_model(model):
    model, _ = model

    assert isinstance(model, torchvision.models.EfficientNet)
    assert model.classifier[1].out_features == 2
    assert len(model.classifier) == 2

    for param in model.features.parameters():
        assert not param.requires_grad

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 2)
