"""
src/model.py

CNN architecture for MNIST handwritten digit classification.

Usage
-----
from model import create_model
model = create_model()  # compiled Keras model
"""

from __future__ import annotations

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build an uncompiled CNN suitable for MNIST.

    Architecture (light but strong for MNIST):
        - Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
        - Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
        - Flatten → Dense(128) → Dropout → Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="mnist_cnn")


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Compile the model with sensible defaults for MNIST."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Convenience factory: build + compile a model.

    Parameters
    ----------
    input_shape : (H, W, C)
    num_classes : number of target classes (MNIST = 10)
    dropout     : dropout rate used in conv blocks
    learning_rate : Adam learning rate
    """
    model = build_model(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    return compile_model(model, learning_rate=learning_rate)


if __name__ == "__main__":
    # Quick local check: prints model summary
    m = create_model()
    m.summary()

