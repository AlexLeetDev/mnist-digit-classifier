"""
serve/app.py
FastAPI backend for MNIST Digit Classifier
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageStat
from tensorflow import keras

app = FastAPI(title="MNIST Digit Classifier API")

# CORS so the /web frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten to your domain when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model.keras"
model = keras.models.load_model(MODEL_PATH)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to model input: (1, 28, 28, 1) float32 in [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    # Invert if background is light so it matches MNIST (white digit on black)
    mean_brightness = float(ImageStat.Stat(img).mean[0])
    if mean_brightness > 127.0:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, ..., None]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns prediction + probabilities.
    """
    content = await file.read()
    x = preprocess(content)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return {"prediction": pred, "probabilities": [float(p) for p in probs]}