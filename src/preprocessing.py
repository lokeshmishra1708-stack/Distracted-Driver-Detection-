"""Safe RGB loading and model-specific tensor normalization."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from config import IMAGE_SIZE


def load_rgb_image(source):
    """Open a local image safely and return an independent, EXIF-oriented RGB image."""
    try:
        with Image.open(source) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("The supplied file is not a valid, supported image.") from exc


def preprocess_image(source, image_size=IMAGE_SIZE):
    """Resize an image and return an RGB CHW float32 tensor scaled to [-1, 1]."""
    image = load_rgb_image(source).resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


def augment_training_image(image, rng=None):
    """Apply cautious, training-only geometric and brightness augmentation."""
    import random
    from PIL import ImageEnhance

    rng = rng or random
    image = image.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
    return image


def image_to_tensor(image, image_size=IMAGE_SIZE):
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
