"""Model loading and prediction without coupling to HTTP routes."""
from pathlib import Path

import torch

from config import CLASS_NAMES, FINAL_MODEL_DIR, IMAGE_SIZE
from src.model import build_model
from src.preprocessing import preprocess_image


class Predictor:
    def __init__(self, model_path=None, model_name="custom_cnn", image_size=IMAGE_SIZE):
        self.model_path = Path(model_path) if model_path else FINAL_MODEL_DIR / "best_model.pt"
        self.model_name, self.image_size = model_name, image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.error = None
        self._load()

    @property
    def available(self):
        return self.model is not None

    def _load(self):
        if not self.model_path.is_file():
            self.error = "No trained model is available yet. Prepare the dataset and train a model first."
            return
        try:
            state = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.model_name = state.get("model_name", self.model_name) if isinstance(state, dict) else self.model_name
            self.image_size = state.get("image_size", self.image_size) if isinstance(state, dict) else self.image_size
            self.model = build_model(self.model_name)
            self.model.load_state_dict(state.get("model_state", state))
            self.model.to(self.device).eval()
        except Exception as exc:
            self.model = None
            self.error = f"The trained model could not be loaded ({type(exc).__name__})."

    def predict(self, source):
        if not self.available:
            raise RuntimeError(self.error or "The trained model is unavailable.")
        batch = preprocess_image(source, self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)[0].cpu().tolist()
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"class_code": list(CLASS_NAMES)[index], "class_name": list(CLASS_NAMES.values())[index], "probability": probabilities[index], "probabilities": [{"code": code, "name": CLASS_NAMES[code], "probability": probabilities[i]} for i, code in enumerate(CLASS_NAMES)]}
