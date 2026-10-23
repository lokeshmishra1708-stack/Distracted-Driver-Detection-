import pandas as pd
import pytest

from src.dataset import grouped_driver_split, validate_manifest


def sample_manifest():
    return pd.DataFrame([{"image_path": f"/tmp/{driver}_{i}.jpg", "class_code": f"c{i % 10}", "label": i % 10, "driver_id": driver, "split": ""} for driver in [f"p{i:02}" for i in range(10)] for i in range(4)])


def test_driver_split_has_no_overlap_and_assigns_all_drivers():
    result = grouped_driver_split(sample_manifest(), seed=48)
    validate_manifest(result)
    splits = [set(result.loc[result.split == name, "driver_id"]) for name in ("train", "validation", "test")]
    assert all(not (splits[i] & splits[j]) for i in range(3) for j in range(i + 1, 3))
    assert set.union(*splits) == set(sample_manifest().driver_id)


def test_invalid_class_mapping_rejected():
    data = sample_manifest()
    data.loc[0, "label"] = 9
    with pytest.raises(ValueError, match="official class-to-index"):
        validate_manifest(data, require_splits=False)


def test_too_few_drivers_has_actionable_error():
    data = sample_manifest().iloc[:4].copy()
    data["driver_id"] = "only-one"
    with pytest.raises(ValueError, match="At least three distinct drivers"):
        grouped_driver_split(data)


def test_manifest_validator_rejects_driver_leakage():
    data = grouped_driver_split(sample_manifest(), seed=48)
    overlapping_driver = data.iloc[0].driver_id
    data.loc[data.split == "validation", "driver_id"] = overlapping_driver
    with pytest.raises(ValueError, match="Driver leakage"):
        validate_manifest(data)


def test_group_split_respects_requested_driver_fraction():
    data = grouped_driver_split(sample_manifest(), seed=48, fractions=(0.70, 0.15, 0.15))
    driver_counts = data.drop_duplicates("driver_id").split.value_counts(normalize=True)
    assert driver_counts["train"] == pytest.approx(0.7, abs=0.11)
