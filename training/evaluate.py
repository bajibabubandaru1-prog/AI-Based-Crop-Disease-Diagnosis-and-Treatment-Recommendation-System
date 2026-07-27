import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES_PATH, DEFAULT_MODEL_NAME, SUPPORTED_MODEL_NAMES, get_model_path
from dataset_loader import load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained crop disease model.")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Trained architecture to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = get_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    _, val_ds, class_names = load_datasets()
    if CLASS_NAMES_PATH.exists():
        class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
