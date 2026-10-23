"""Predict a single JPG or PNG image using the latest trained model."""
import argparse
import json

from src.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--model", choices=("custom_cnn", "mobilenetv2"), default="custom_cnn")
    args = parser.parse_args()
    predictor = Predictor(args.model_path, args.model)
    if not predictor.available:
        parser.error(predictor.error)
    try:
        print(json.dumps(predictor.predict(args.image), indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
