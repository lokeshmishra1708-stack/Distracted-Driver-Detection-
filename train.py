"""Train a custom CNN or MobileNetV2 using driver-grouped splits."""
import argparse
import logging

from config import IMAGE_SIZE, SPLITS_DIR
from src.dataset import validate_manifest
from src.model import build_model
from src.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("custom_cnn", "mobilenetv2"), default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2, help="Limit PyTorch CPU compute threads to keep the machine responsive")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use MobileNetV2 ImageNet weights; requires working torchvision and network access if uncached")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze the final MobileNetV2 feature blocks")
    parser.add_argument("--resume", help="Resume model and optimizer state from a project .pt checkpoint")
    parser.add_argument("--manifest", default=str(SPLITS_DIR / "manifest.csv"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.early_stopping_patience < 1 or args.cpu_threads < 1:
        parser.error("epochs, batch-size, early-stopping-patience, and cpu-threads must be positive; image-size must be at least 32")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import pandas as pd
    manifest = pd.read_csv(args.manifest)
    validate_manifest(manifest)
    model = build_model(args.model, pretrained=args.pretrained)
    if args.fine_tune and args.model == "mobilenetv2":
        model.unfreeze_tail()
    result = train_model(args.manifest, model, model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size, smoke_test=args.smoke_test, workers=args.workers, learning_rate=args.learning_rate, resume=args.resume, early_stopping_patience=args.early_stopping_patience, cpu_threads=args.cpu_threads)
    print(f"Device: {result['device']}\nBest checkpoint: {result['checkpoint']}\nInference model: {result['final_model'] or 'not created (smoke test only)'}\nTraining history: {result['history']}\nTraining metrics CSV: {result['history_csv']}")


if __name__ == "__main__":
    main()
