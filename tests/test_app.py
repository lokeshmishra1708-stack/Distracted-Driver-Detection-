import io

import pytest
from PIL import Image

from app import create_app


class FakePredictor:
    available = True

    def predict(self, _stream):
        probabilities = [{"code": f"c{i}", "name": f"Class {i}", "probability": 0.1} for i in range(10)]
        return {"class_code": "c0", "class_name": "Safe driving", "probability": 0.1, "probabilities": probabilities}


def png_upload():
    image = io.BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")
    image.seek(0)
    return image


def test_home_and_health_without_model():
    class Missing:
        available = False
        error = "missing"
    client = create_app({"TESTING": True}, Missing()).test_client()
    assert client.get("/").status_code == 200
    assert b"SETUP REQUIRED" in client.get("/").data
    assert client.get("/health").json["model_available"] is False
    assert client.post("/predict").status_code == 503


def test_prediction_route_and_invalid_upload():
    client = create_app({"TESTING": True}, FakePredictor()).test_client()
    response = client.post("/predict", data={"image": (png_upload(), "frame.png")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["class_code"] == "c0"
    bad = client.post("/predict", data={"image": (io.BytesIO(b"bad"), "frame.exe")}, content_type="multipart/form-data")
    assert bad.status_code == 400


def test_corrupt_upload_and_history_clear():
    client = create_app({"TESTING": True}, FakePredictor()).test_client()
    bad = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "frame.png")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    client.post("/predict", data={"image": (png_upload(), "frame.png")}, content_type="multipart/form-data")
    assert client.post("/history/clear").json["cleared"] is True
