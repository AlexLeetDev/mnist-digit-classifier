"""
src/inference.py

Load a trained MNIST model and run inference on new images.
Useful for testing predictions or integrating with other apps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat
from tensorflow import keras


def load_model(model_path: Path = Path("outputs/model.keras")) -> keras.Model:
    """Load a trained Keras model from disk."""
    print(f"Loading model from {model_path} ...")
    model = keras.models.load_model(model_path)
    return model


def _maybe_invert(img: Image.Image) -> Image.Image:
    mean_brightness: float = float(ImageStat.Stat(img).mean[0])
    return ImageOps.invert(img) if mean_brightness > 127.0 else img


def preprocess_image(img_path: Path) -> np.ndarray:
    """Convert an image to model-ready tensor shaped (1, 28, 28, 1)."""
    img: Image.Image = Image.open(img_path).convert("L")  # grayscale
    img = _maybe_invert(img)
    img = img.resize((28, 28))
    arr: np.ndarray = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None, ..., None]  # (1, 28, 28, 1)
    return arr


def predict(model: keras.Model, x: np.ndarray) -> Tuple[int, np.ndarray]:
    """Run inference and return (predicted_label, probabilities)."""
    probs: np.ndarray = model.predict(x, verbose=0)[0]
    pred: int = int(np.argmax(probs))
    return pred, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on an MNIST image")
    parser.add_argument("image", type=str, help="Path to 28x28 digit image (png/jpg)")
    parser.add_argument("--model", type=str, default="outputs/model.keras")
    args = parser.parse_args()

    model = load_model(Path(args.model))
    x = preprocess_image(Path(args.image))
    pred, probs = predict(model, x)

    print(f"Prediction: {pred}")
    print("Probabilities:", np.round(probs, 3))


if __name__ == "__main__":
    main()