import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    DEFAULT_MODEL_NAME,
    IMAGE_SIZE,
    SUPPORTED_MODEL_NAMES,
    get_model_path,
)


def predict_image(image_path: Path, model_name: str = DEFAULT_MODEL_NAME):
    model_path = get_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    model = tf.keras.models.load_model(model_path)

    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    return {
        "class_name": class_names[predicted_index],
        "confidence": float(predictions[predicted_index]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Trained architecture to use for prediction.",
    )
    args = parser.parse_args()

    result = predict_image(args.image_path, model_name=args.model)
    print(f"Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
