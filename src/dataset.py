"""Dataset discovery, verified driver mapping, and reproducible grouped splits."""
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
from src.preprocessing import augment_training_image, image_to_tensor

REQUIRED_MANIFEST_COLUMNS = {"image_path", "class_code", "label", "driver_id", "split"}
SPLIT_NAMES = ("train", "validation", "test")


def discover_dataset(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}. Download and extract the competition files first.")
    metadata = list(root.rglob("driver_imgs_list.csv"))
    images = [p for p in root.rglob("*.jpg") if p.parent.name in CLASS_NAMES]
    if not metadata:
        raise FileNotFoundError("driver_imgs_list.csv was not found. Extract Kaggle's labelled competition archive, including its driver metadata.")
    if not images:
        raise FileNotFoundError("No labelled class/c0..c9/*.jpg images were found. Extract the labelled training archive and retry.")
    metadata.sort(key=lambda p: ("driver_imgs_list.csv" not in p.as_posix().lower(), len(p.parts)))
    return images, metadata[0]


def make_manifest(image_paths, metadata_path):
    """Map each image to its official driver/class metadata; fail on inconsistencies."""
    table = pd.read_csv(metadata_path, dtype=str)
    required = {"subject", "classname", "img"}
    if not required.issubset(table.columns):
        raise ValueError(f"Driver metadata must have columns {sorted(required)}; found {list(table.columns)}")
    if table[list(required)].isna().any().any() or table.empty:
        raise ValueError("Driver metadata has missing values in required columns.")
    table["class_code"] = table.classname.map(lambda value: value if value in CLASS_NAMES else None)
    if table.class_code.isna().any():
        bad = sorted(table.loc[table.class_code.isna(), "classname"].unique())
        raise ValueError(f"Unrecognized classes in official driver metadata: {bad}")
    keys = table.assign(img_key=table.img.map(lambda value: Path(value).name.lower()))
    if keys.img_key.duplicated().any():
        raise ValueError("Driver metadata contains ambiguous duplicate image filenames.")
    by_name = keys.set_index("img_key").to_dict("index")
    rows = []
    found_names = set()
    for image_path in image_paths:
        image_path = Path(image_path).resolve()
        code = image_path.parent.name
        if code not in CLASS_NAMES:
            raise ValueError(f"Image is outside a recognized class folder: {image_path}")
        key = image_path.name.lower()
        record = by_name.get(key)
        if record is None:
            raise ValueError(f"Image {image_path.name} cannot be reliably mapped to driver_imgs_list.csv.")
        if record["class_code"] != code:
            raise ValueError(f"Class mismatch for {image_path.name}: folder says {code}, metadata says {record['class_code']}.")
        if not image_path.is_file():
            raise ValueError(f"Dataset image is missing: {image_path}")
        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception as exc:
            raise ValueError(f"Corrupt or unreadable image {image_path}: {exc}") from exc
        found_names.add(key)
        rows.append({"image_path": str(image_path), "class_code": code, "label": list(CLASS_NAMES).index(code), "driver_id": str(record["subject"]).strip(), "split": ""})
    if len(found_names) != len(by_name):
        missing = set(by_name).difference(found_names)
        if missing:
            raise ValueError(f"Metadata refers to {len(missing)} images not present in the discovered labelled folders (example: {next(iter(missing))}).")
    manifest = pd.DataFrame(rows)
    if manifest.driver_id.eq("").any():
        raise ValueError("Official metadata contains empty driver identifiers.")
    return manifest


def grouped_driver_split(manifest, seed=RANDOM_SEED, fractions=(0.70, 0.15, 0.15)):
    """Assign each subject to exactly one partition; reduce split count if driver count requires."""
    validate_manifest(manifest, require_splits=False)
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Split fractions must be three positive numbers summing to one.")
    result = manifest.copy()
    drivers = result.driver_id.nunique()
    if drivers < 3:
        raise ValueError(f"At least three distinct drivers are required for train/validation/test; found {drivers}.")
    groups = result[["driver_id"]].drop_duplicates().reset_index(drop=True)
    holdout = fractions[1] + fractions[2]
    first = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_ix, remaining_ix = next(first.split(groups, groups=groups.driver_id))
    train_drivers = set(groups.iloc[train_ix].driver_id)
    remaining = groups.iloc[remaining_ix]
    val_fraction = fractions[1] / (fractions[1] + fractions[2])
    second = GroupShuffleSplit(n_splits=1, test_size=1 - val_fraction, random_state=seed + 1)
    val_ix, test_ix = next(second.split(remaining, groups=remaining.driver_id))
    assignment = {driver: "train" for driver in train_drivers}
    assignment.update({driver: "validation" for driver in remaining.iloc[val_ix].driver_id})
    assignment.update({driver: "test" for driver in remaining.iloc[test_ix].driver_id})
    result["split"] = result.driver_id.map(assignment)
    validate_manifest(result)
    return result


def validate_manifest(manifest, require_splits=True):
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if manifest.empty or manifest[list(REQUIRED_MANIFEST_COLUMNS - {"split"} if not require_splits else REQUIRED_MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError("Manifest must contain non-empty, complete image, class, label, and driver data.")
    if not set(manifest.class_code).issubset(CLASS_NAMES):
        raise ValueError("Manifest contains unknown official class codes.")
    if any(int(row.label) != list(CLASS_NAMES).index(row.class_code) for row in manifest.itertuples()):
        raise ValueError("Manifest labels do not match the official class-to-index mapping.")
    if require_splits:
        if not set(manifest.split).issubset(SPLIT_NAMES) or set(manifest.split) != set(SPLIT_NAMES):
            raise ValueError("Manifest must contain train, validation, and test split assignments.")
        driver_sets = {split: set(manifest.loc[manifest.split == split, "driver_id"]) for split in SPLIT_NAMES}
        for i, first in enumerate(SPLIT_NAMES):
            for second in SPLIT_NAMES[i + 1:]:
                overlap = driver_sets[first] & driver_sets[second]
                if overlap:
                    raise ValueError(f"Driver leakage: {sorted(overlap)} occur in both {first} and {second}.")
    return True


def describe_splits(manifest):
    return {split: {"images": int((manifest.split == split).sum()), "drivers": int(manifest.loc[manifest.split == split, "driver_id"].nunique()), "classes": {code: int(Counter(manifest.loc[manifest.split == split, "class_code"]).get(code, 0)) for code in CLASS_NAMES}} for split in SPLIT_NAMES}


class DriverImageDataset(Dataset):
    def __init__(self, manifest, split, image_size=IMAGE_SIZE, augment=False):
        self.rows = manifest.loc[manifest.split == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No examples found for split {split!r}.")
        self.image_size, self.augment = image_size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            if self.augment:
                image = augment_training_image(image)
            tensor = image_to_tensor(image, self.image_size)
        return tensor, int(row.label)
