"""Shared project configuration and official competition class mapping."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
FINAL_MODEL_DIR = ROOT / "models" / "final"
IMAGE_SIZE = 160
NUM_CLASSES = 10
CLASS_NAMES = {
    "c0": "Safe driving",
    "c1": "Texting with right hand",
    "c2": "Talking on phone with right hand",
    "c3": "Texting with left hand",
    "c4": "Talking on phone with left hand",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger",
}
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CODE = list(CLASS_NAMES)
RANDOM_SEED = 48
UPLOAD_MAX_MB = 8
