import argparse

import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    DEFAULT_MODEL_NAME,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAST_LAYERS,
    MODEL_DIR,
    SUPPORTED_MODEL_NAMES,
    get_model_path,
)
from dataset_loader import load_datasets
from model import build_transfer_learning_model, unfreeze_for_fine_tuning


def parse_args():
    parser = argparse.ArgumentParser(description="Train a crop disease classifier.")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Transfer learning architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Frozen feature-extraction training epochs.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help="Fine-tuning epochs after unfreezing the final base-model layers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = get_model_path(args.model)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    model = build_transfer_learning_model(
        model_name=args.model,
        num_classes=len(class_names),
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            # monitor="val_loss",
            # patience=3,
            # restore_best_weights=True,
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        print(
            "Starting fine-tuning: "
            f"unfreezing last {FINE_TUNE_LAST_LAYERS} {args.model} layers"
        )
        model = unfreeze_for_fine_tuning(
            model,
            model_name=args.model,
            last_layers=FINE_TUNE_LAST_LAYERS,
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=args.epochs,
            epochs=args.epochs + args.fine_tune_epochs,
            callbacks=callbacks,
        )
        history.history["accuracy"].extend(fine_tune_history.history["accuracy"])
        history.history["val_accuracy"].extend(fine_tune_history.history["val_accuracy"])

    model.save(model_path)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
