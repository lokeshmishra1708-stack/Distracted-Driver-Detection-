"""Streaming PyTorch training loop with best-checkpoint selection and smoke mode."""
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from config import CHECKPOINT_DIR, FINAL_MODEL_DIR, IMAGE_SIZE, RANDOM_SEED
from src.dataset import DriverImageDataset
from src.utils import get_device, seed_everything

LOGGER = logging.getLogger("distracted_driver")


def train_model(manifest_path, model, model_name="custom_cnn", epochs=10, batch_size=8, image_size=IMAGE_SIZE, smoke_test=False, workers=0, learning_rate=1e-3, resume=None, early_stopping_patience=5, cpu_threads=2):
    seed_everything(RANDOM_SEED)
    manifest = pd.read_csv(manifest_path)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(max(1, cpu_threads))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
    train_data = DriverImageDataset(manifest, "train", image_size, augment=True)
    val_data = DriverImageDataset(manifest, "validation", image_size, augment=False)
    if smoke_test:
        train_data = Subset(train_data, range(min(4, len(train_data))))
        val_data = Subset(val_data, range(min(2, len(val_data))))
        epochs, batch_size = 1, min(batch_size, 2)
    loader_args = {"batch_size": batch_size, "num_workers": max(0, workers), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_suffix = "_smoke" if smoke_test else ""
    best_path = checkpoint_dir / f"{model_name}{artifact_suffix}_best.pt"
    state = None
    if resume:
        resume = Path(resume)
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state", state))
        if isinstance(state, dict) and "model_state" in state:
            best_path = resume
        if state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        LOGGER.info("Resumed model state from %s", resume)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    history = state.get("history", []) if isinstance(state, dict) else []
    best_accuracy = state.get("best_val_accuracy", -1.0) if isinstance(state, dict) else -1.0
    start_epoch = state.get("epoch", 0) if isinstance(state, dict) else 0
    epochs_without_improvement = 0
    LOGGER.info("Training %s on %s (%d training images, batch_size=%d)", model_name, device, len(train_data), batch_size)
    for epoch in range(start_epoch, epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if not smoke_test and (batch_index + 1) % 100 == 0:
                LOGGER.info("Epoch %d/%d: processed %d/%d training batches", epoch + 1, epochs, batch_index + 1, len(train_loader))
            if smoke_test and batch_index >= 1:
                break
        train_loss, train_accuracy = loss_sum / max(total, 1), correct / max(total, 1)
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss_sum += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                if not smoke_test and (batch_index + 1) % 100 == 0:
                    LOGGER.info("Epoch %d/%d: processed %d/%d validation batches", epoch + 1, epochs, batch_index + 1, len(val_loader))
                if smoke_test and batch_index >= 0:
                    break
        val_loss, val_accuracy = val_loss_sum / max(val_total, 1), val_correct / max(val_total, 1)
        scheduler.step(val_accuracy)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        LOGGER.info("Epoch %d/%d train loss %.4f accuracy %.4f | val loss %.4f accuracy %.4f", epoch + 1, epochs, train_loss, train_accuracy, val_loss, val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "model_name": model_name, "image_size": image_size, "best_val_accuracy": best_accuracy, "epoch": epoch + 1, "history": history}, best_path)
        else:
            epochs_without_improvement += 1
        if not smoke_test and epochs_without_improvement >= early_stopping_patience:
            LOGGER.info("Early stopping after %d epochs without validation accuracy improvement", epochs_without_improvement)
            break
    final_path = None
    if not smoke_test:
        final_dir = FINAL_MODEL_DIR
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "best_model.pt"
        shutil.copy2(best_path, final_path)
    history_path = Path("outputs/reports") / f"{model_name}{artifact_suffix}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    history_csv = Path("outputs/reports") / f"{model_name}{artifact_suffix}_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    LOGGER.info("Smoke-test checkpoint: %s; full-training deployable model: %s; history: %s", best_path, final_path, history_path)
    return {"checkpoint": best_path, "final_model": final_path, "history": history_path, "history_csv": history_csv, "device": str(device), "model": model}
