import tensorflow as tf

from config import CLASS_NAMES_PATH, EPOCHS, MODEL_DIR, MODEL_PATH
from dataset_loader import load_datasets
from model import build_mobilenetv2_model


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
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
