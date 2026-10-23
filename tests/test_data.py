import pandas as pd
import pytest
from PIL import Image

from config import CLASS_NAMES
from src.dataset import grouped_driver_split, make_manifest
from src.preprocessing import load_rgb_image, preprocess_image


def test_official_class_mapping():
    assert list(CLASS_NAMES) == [f"c{i}" for i in range(10)]
    assert CLASS_NAMES["c0"] == "Safe driving"
    assert CLASS_NAMES["c9"] == "Talking to passenger"


def test_preprocessing_rgb_expected_shape(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (80, 50), (100, 130, 150)).save(path)
    tensor = preprocess_image(path, 64)
    assert tuple(tensor.shape) == (3, 64, 64)
    assert tensor.dtype.name if hasattr(tensor.dtype, "name") else str(tensor.dtype) == "torch.float32"
    assert -1 <= tensor.min() <= tensor.max() <= 1


def test_invalid_image_fails_clearly(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="valid, supported image"):
        load_rgb_image(path)


def test_manifest_maps_official_driver_and_class(tmp_path):
    image = tmp_path / "c0" / "img_1.jpg"
    image.parent.mkdir()
    Image.new("RGB", (16, 16)).save(image)
    metadata = tmp_path / "driver_imgs_list.csv"
    pd.DataFrame([{"subject": "p001", "classname": "c0", "img": "img_1.jpg"}]).to_csv(metadata, index=False)
    manifest = make_manifest([image], metadata)
    assert manifest.iloc[0].driver_id == "p001"
    assert manifest.iloc[0].label == 0


def test_manifest_fails_on_bad_driver_mapping(tmp_path):
    image = tmp_path / "c1" / "img_1.jpg"
    image.parent.mkdir()
    Image.new("RGB", (16, 16)).save(image)
    metadata = tmp_path / "driver_imgs_list.csv"
    pd.DataFrame([{"subject": "p001", "classname": "c0", "img": "img_1.jpg"}]).to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="Class mismatch"):
        make_manifest([image], metadata)


def test_missing_image_rejected(tmp_path):
    with pytest.raises(ValueError, match="valid, supported image"):
        load_rgb_image(tmp_path / "nope.jpg")
