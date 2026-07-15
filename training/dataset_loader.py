from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, IMAGE_SIZE, SEED, TRAIN_DIR, VAL_DIR


def validate_dataset_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {path}\n"
            "Extract archive.zip so the dataset contains PlantVillage/train and PlantVillage/val."
        )

    class_dirs = [item for item in path.iterdir() if item.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found inside: {path}")


def load_datasets():
    validate_dataset_path(TRAIN_DIR)
    validate_dataset_path(VAL_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names
