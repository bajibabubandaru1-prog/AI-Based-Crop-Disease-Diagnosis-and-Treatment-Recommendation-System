import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH


def predict_image(image_path: Path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    model = tf.keras.models.load_model(MODEL_PATH)

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
    args = parser.parse_args()

    result = predict_image(args.image_path)
    print(f"Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
