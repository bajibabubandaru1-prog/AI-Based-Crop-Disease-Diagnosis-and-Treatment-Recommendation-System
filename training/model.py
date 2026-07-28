import tensorflow as tf

from config import FINE_TUNE_LEARNING_RATE, IMAGE_SIZE, LEARNING_RATE
from preprocess import build_augmentation_layer


def build_transfer_learning_model(
    model_name: str,
    num_classes: int,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = build_augmentation_layer()(inputs)

    if model_name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights=weights,
            name="mobilenetv2_base",
        )
    elif model_name == "efficientnetb0":
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights=weights,
            name="efficientnetb0_base",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base_model.trainable = weights is None

    x = base_model(x, training=weights is None)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"crop_disease_{model_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    model_name: str,
    last_layers: int,
) -> tf.keras.Model:
    base_model = model.get_layer(f"{model_name}_base")
    base_model.trainable = True

    for layer in base_model.layers[:-last_layers]:
        layer.trainable = False

    # Keep BatchNorm stable during fine-tuning on a limited dataset.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
