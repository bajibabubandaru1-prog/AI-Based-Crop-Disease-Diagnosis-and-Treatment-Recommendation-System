from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_NAME = "mobilenetv2"
MODEL_FILENAMES = {
    "mobilenetv2": "crop_disease_mobilenetv2.keras",
    "efficientnetb0": "crop_disease_efficientnetb0.keras",
}


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_FILENAMES:
        supported = ", ".join(MODEL_FILENAMES)
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {supported}")
    return MODEL_DIR / MODEL_FILENAMES[model_name]


@lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL_NAME):
    model_path = get_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    return CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()


def predict_leaf_image(image_path: Path, model_name: str = DEFAULT_MODEL_NAME, top_k: int = 3):
    model = load_model(model_name)
    class_names = load_class_names()

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    predictions = [
        {
            "class_name": class_names[index],
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]

    return {
        "model_name": model_name,
        "top_prediction": predictions[0],
        "top_predictions": predictions,
    }
