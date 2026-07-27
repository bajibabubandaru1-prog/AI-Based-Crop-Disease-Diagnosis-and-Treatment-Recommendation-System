import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAST_LAYERS,
    MODEL_DIR,
    MODEL_PATH,
)
from dataset_loader import load_datasets
from model import build_mobilenetv2_model, unfreeze_for_fine_tuning


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    model = build_mobilenetv2_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
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
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    if FINE_TUNE_EPOCHS > 0:
        print(
            "Starting fine-tuning: "
            f"unfreezing last {FINE_TUNE_LAST_LAYERS} MobileNetV2 layers"
        )
        model = unfreeze_for_fine_tuning(
            model,
            last_layers=FINE_TUNE_LAST_LAYERS,
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=EPOCHS,
            epochs=EPOCHS + FINE_TUNE_EPOCHS,
            callbacks=callbacks,
        )
        history.history["accuracy"].extend(fine_tune_history.history["accuracy"])
        history.history["val_accuracy"].extend(fine_tune_history.history["val_accuracy"])

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
