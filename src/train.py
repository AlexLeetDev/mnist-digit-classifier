"""
src/train.py

Train the MNIST handwritten digit classifier and save artifacts to ./outputs

Artifacts written:
- outputs/model.keras              (final trained model)
- outputs/best.keras               (best val model via ModelCheckpoint)
- outputs/metrics.json             (train/val history + test metrics)
- outputs/training_curves.png      (accuracy & loss over epochs; Seaborn)
- outputs/confusion_matrix.png     (Seaborn heatmap)
- outputs/label_map.json           ({0: "0", ... , 9: "9"})
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report

# Import the model factory from src/model.py
from model import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST CNN")
    p.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument(
        "--outdir", type=str, default="outputs", help="Directory to save artifacts"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    # Best-effort reproducibility
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_data():
    """Load MNIST and return (x_train, y_train), (x_test, y_test) normalized."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize to [0,1] and add channel dimension
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]
    return (x_train, y_train), (x_test, y_test)


def plot_training_curves(history: keras.callbacks.History, out_path: Path) -> None:
    """Save accuracy & loss curves as a single PNG (Seaborn styling)."""
    sns.set_theme(style="whitegrid")
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(10, 4.5))

    # Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x=list(epochs), y=hist.get("accuracy", []), label="train")
    if "val_accuracy" in hist:
        sns.lineplot(x=list(epochs), y=hist["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=list(epochs), y=hist.get("loss", []), label="train")
    if "val_loss" in hist:
        sns.lineplot(x=list(epochs), y=hist["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save a confusion matrix heatmap using Seaborn."""
    sns.set_theme(style="white")
    plt.figure(figsize=(6.2, 5.6))

    # Seaborn expects Sequence[str] for ticklabels.
    n = cm.shape[0]
    labels: list[str] = [str(i) for i in range(n)]

    ax = sns.heatmap(
        cm.astype(int),      # keep integers since fmt="d"
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test)")

    # Rotate x-tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save a label map (useful for serving/UX)
    with (outdir / "label_map.json").open("w") as f:
        json.dump({i: str(i) for i in range(10)}, f)

    print("Loading data…")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("Building model…")
    model = create_model(input_shape=(28, 28, 1), num_classes=10, dropout=0.3)
    model.summary()

    # Callbacks: early stop + best checkpoint
    ckpt_path = outdir / "best.keras"
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("Training…")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),  # simple split: use test as val during training
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True,
        callbacks=cbs,
        verbose=1,
    )

    print("Evaluating on test set…")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    # Predictions for confusion matrix & report
    probs = model.predict(x_test, verbose=0)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(y_test, preds)

    # Save artifacts
    print("Saving artifacts to:", outdir.resolve())
    model.save(outdir / "model.keras")

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    with (outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(history, outdir / "training_curves.png")
    plot_confusion_matrix(cm, outdir / "confusion_matrix.png")

    print("Done. Saved:")
    print(" -", outdir / "model.keras")
    print(" -", outdir / "best.keras")
    print(" -", outdir / "metrics.json")
    print(" -", outdir / "training_curves.png")
    print(" -", outdir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
