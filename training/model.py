import tensorflow as tf

from config import FINE_TUNE_LEARNING_RATE, IMAGE_SIZE, LEARNING_RATE
from preprocess import build_augmentation_layer


def build_mobilenetv2_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = build_augmentation_layer()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    last_layers: int,
) -> tf.keras.Model:
    base_model = model.get_layer("mobilenetv2_base")
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
