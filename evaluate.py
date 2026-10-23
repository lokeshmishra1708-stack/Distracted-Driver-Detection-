"""Evaluate one selected checkpoint on the driver-held-out test split only."""
import argparse
import json

from config import IMAGE_SIZE, SPLITS_DIR
from src.evaluation import evaluate_model, write_history_plot
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", choices=("custom_cnn", "mobilenetv2"), default="custom_cnn")
    parser.add_argument("--manifest", default=str(SPLITS_DIR / "manifest.csv"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    model = build_model(args.model)
    metrics = evaluate_model(model, args.manifest, args.model_path, args.image_size, args.batch_size)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))
    history_path = f"outputs/reports/{args.model}_history.json"
    try:
        write_history_plot(history_path, args.model)
    except FileNotFoundError:
        print(f"Training history missing; skipped curves: {history_path}")
    print("Reports: outputs/reports/evaluation.json, outputs/reports/classification_report.txt")
    print("Charts: outputs/figures/confusion_matrix.png, outputs/figures/per_class_f1.png")


if __name__ == "__main__":
    main()
