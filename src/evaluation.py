"""Driver-held-out test evaluation, reports, and model resource measurements."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from config import CLASS_NAMES, IMAGE_SIZE
from src.dataset import DriverImageDataset, describe_splits
from src.visualization import plot_class_f1, plot_confusion_matrix, plot_history


def evaluate_model(model, manifest_path, checkpoint_path, image_size=IMAGE_SIZE, batch_size=8):
    manifest = pd.read_csv(manifest_path)
    test_set = DriverImageDataset(manifest, "test", image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    y_true, y_pred, probabilities = [], [], []
    loss_sum, total, latencies = 0.0, 0, []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) / len(labels))
            loss_sum += criterion(logits, labels).item() * len(labels)
            total += len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
    names = list(CLASS_NAMES.values())
    labels = list(range(len(names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    scores = {"model_name": state.get("model_name", "custom_cnn"), "test_loss": loss_sum / max(1, total), "test_accuracy": accuracy_score(y_true, y_pred), "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "per_class": report, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "split_statistics": describe_splits(manifest), "model_parameters": sum(p.numel() for p in model.parameters()), "parameter_storage_bytes": sum(p.numel() * p.element_size() for p in model.parameters()), "checkpoint_file_bytes": Path(checkpoint_path).stat().st_size, "inference_latency_ms_per_image": float(np.mean(latencies) * 1000), "device": str(device), "test_images": total}
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    text_report = classification_report(y_true, y_pred, labels=labels, target_names=[f"{code}: {name}" for code, name in CLASS_NAMES.items()], zero_division=0)
    (out / "classification_report.txt").write_text(text_report, encoding="utf-8")
    figures = Path("outputs/figures")
    plot_confusion_matrix(scores["confusion_matrix"], list(CLASS_NAMES), figures / "confusion_matrix.png")
    plot_class_f1(report, figures / "per_class_f1.png")
    return scores


def write_history_plot(history_path, model_name):
    history = pd.read_json(history_path)
    plot_history(history, "outputs/figures", model_name)
