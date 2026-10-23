"""Responsive distracted-driver classification dashboard."""
import logging
import os
import secrets
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, session
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import CLASS_NAMES, FINAL_MODEL_DIR, UPLOAD_MAX_MB
from src.inference import Predictor

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def create_app(test_config=None, predictor=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=UPLOAD_MAX_MB * 1024 * 1024, SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32))
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chosen_path = app.config.get("MODEL_PATH") or os.environ.get("MODEL_PATH", str(FINAL_MODEL_DIR / "best_model.pt"))
    app.extensions["predictor"] = predictor if predictor is not None else Predictor(chosen_path, os.environ.get("MODEL_NAME", "custom_cnn"))

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="Image exceeds the configured upload size limit."), 413

    @app.get("/")
    def index():
        model = app.extensions["predictor"]
        recent = session.get("recent_predictions", [])
        stats_path = Path("data/processed/dataset_stats.json")
        dataset_stats = None
        if stats_path.is_file():
            try:
                import json
                dataset_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logging.warning("Dataset statistics are unreadable; dashboard will show dataset status without counts.")
        return render_template("index.html", model_available=model.available, model_status=("Ready" if model.available else "Unavailable — model not trained"), model_message=("Predictions are experimental and not a safety system." if model.available else "Train a model before uploading images. Kaggle dataset and driver-grouped splits are required."), recent=recent, class_count=len(CLASS_NAMES), dataset_stats=dataset_stats)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_available=app.extensions["predictor"].available)

    @app.get("/metrics")
    def metrics():
        path = Path("outputs/reports/evaluation.json")
        if not path.is_file():
            return jsonify(available=False, message="Evaluation metrics are not available until a held-out evaluation has completed."), 200
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("per_class", None)
        payload.pop("confusion_matrix", None)
        return jsonify(available=True, metrics=payload)

    @app.get("/figures/<name>")
    def figure(name):
        allowed = {"confusion_matrix.png", "per_class_f1.png", "custom_cnn_accuracy.png", "custom_cnn_loss.png", "mobilenetv2_accuracy.png", "mobilenetv2_loss.png"}
        if name not in allowed:
            abort(404)
        return send_from_directory(Path("outputs/figures"), name)

    @app.post("/predict")
    def predict():
        model = app.extensions["predictor"]
        if not model.available:
            return jsonify(error="Predictions are unavailable: train a model first."), 503
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a JPEG or PNG image to upload."), 400
        filename = secure_filename(upload.filename)
        if "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify(error="Unsupported file type. Upload a JPG, JPEG, or PNG image."), 400
        try:
            upload.stream.seek(0)
            with Image.open(upload.stream) as image:
                image.verify()
            upload.stream.seek(0)
            result = model.predict(upload.stream)
        except (OSError, ValueError) as exc:
            logging.info("Rejected invalid uploaded image (%s)", type(exc).__name__)
            return jsonify(error="The selected file is not a valid image."), 400
        except RuntimeError:
            return jsonify(error="The trained model is temporarily unavailable."), 503
        recent = session.get("recent_predictions", [])
        recent.insert(0, {"filename": filename, "class_code": result["class_code"], "class_name": result["class_name"]})
        session["recent_predictions"] = recent[:10]
        logging.info("Prediction completed for an uploaded image (filename omitted from logs)")
        return jsonify(result)

    @app.post("/history/clear")
    def clear_history():
        session.pop("recent_predictions", None)
        return jsonify(cleared=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
