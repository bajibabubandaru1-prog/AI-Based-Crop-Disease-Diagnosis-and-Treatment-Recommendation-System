"""Model discovery, loading, and leaf-image prediction utilities."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

try:
    from .config import PROJECT_ROOT
except ImportError:  # Allows: python backend/app.py
    from config import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_NAME = "mobilenetv2"
MODEL_FILENAMES = {
    "mobilenetv2": "crop_disease_mobilenetv2.keras",
    "efficientnetb0": "crop_disease_efficientnetb0.keras",
}
MODEL_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_FILENAMES:
        raise ValueError("Choose a supported prediction model.")
    return MODEL_DIR / MODEL_FILENAMES[model_name]


def get_available_models() -> list[dict]:
    """Return model metadata without attempting to load TensorFlow models."""
    return [
        {
            "id": name,
            "label": MODEL_LABELS[name],
            "available": get_model_path(name).exists(),
        }
        for name in MODEL_FILENAMES
    ]


@lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL_NAME):
    model_path = get_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"The {MODEL_LABELS[model_name]} model has not been trained yet. "
            "Train it first or choose an available model."
        )
    return tf.keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("The class names file is missing from the models folder.")
    return [name for name in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines() if name]


def validate_image(image_path: Path) -> None:
    """Verify the uploaded file is a readable image before TensorFlow loads it."""
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def format_class_name(class_name: str) -> str:
    return class_name.replace("___", " - ").replace("_", " ")


def predict_leaf_image(image_path: Path, model_name: str = DEFAULT_MODEL_NAME, top_k: int = 3) -> dict:
    validate_image(image_path)
    model = load_model(model_name)
    class_names = load_class_names()

    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise RuntimeError("Model output does not match the configured disease classes.")

    safe_top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[-safe_top_k:][::-1]
    predictions = [
        {
            "class_name": class_names[index],
            "display_name": format_class_name(class_names[index]),
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]

    return {
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "top_prediction": predictions[0],
        "top_predictions": predictions,
    }
