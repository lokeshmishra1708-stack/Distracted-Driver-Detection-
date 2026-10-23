import torch

from src.model import CustomCNN


def test_custom_cnn_output_shape_and_probabilities():
    model = CustomCNN().eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 64, 64))
        probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (2, 10)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(1), torch.ones(2), atol=1e-6)


def test_trainable_model_has_parameters():
    assert sum(p.numel() for p in CustomCNN().parameters()) > 0
