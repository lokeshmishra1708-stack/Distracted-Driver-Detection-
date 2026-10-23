"""Save publication-friendly training curves and per-class evaluation charts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, output_dir, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history[f"train_{metric}"], label="Training")
        ax.plot(history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{model_name} {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_name}_{metric}.png", dpi=140)
        plt.close(fig)


def plot_confusion_matrix(matrix, labels, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual", title="Driver-held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_class_f1(report, output_path):
    labels = list(report)
    labels = [label for label in labels if label.startswith("c")]
    scores = [report[label]["f1-score"] for label in labels]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, scores, color="#20b8d6")
    ax.set(ylim=(0, 1), ylabel="F1-score", title="Per-class test F1-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
