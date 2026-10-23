import io

import pytest
import torch
from PIL import Image

from src.inference import Predictor
from src.model import CustomCNN


def test_missing_model_unavailable(tmp_path):
    predictor = Predictor(tmp_path / "missing.pt")
    assert not predictor.available
    with pytest.raises(RuntimeError, match="No trained model"):
        predictor.predict(io.BytesIO(b""))


def test_predict_returns_normalized_probabilities(tmp_path):
    path = tmp_path / "tiny.pt"
    torch.save({"model_state": CustomCNN().state_dict(), "model_name": "custom_cnn", "image_size": 32}, path)
    predictor = Predictor(path)
    image = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 100, 200)).save(image, format="PNG")
    image.seek(0)
    result = predictor.predict(image)
    assert result["class_code"] in [f"c{i}" for i in range(10)]
    assert len(result["probabilities"]) == 10
    assert sum(item["probability"] for item in result["probabilities"]) == pytest.approx(1.0)
