from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "plantvillage" / "archive" / "PlantVillage"
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAMES = {
    "mobilenetv2": "crop_disease_mobilenetv2.keras",
    "efficientnetb0": "crop_disease_efficientnetb0.keras",
}
DEFAULT_MODEL_NAME = "mobilenetv2"
SUPPORTED_MODEL_NAMES = tuple(MODEL_FILENAMES)
MODEL_PATH = MODEL_DIR / MODEL_FILENAMES[DEFAULT_MODEL_NAME]
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LEARNING_RATE = 0.00001
FINE_TUNE_LAST_LAYERS = 30
SEED = 42


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_FILENAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {supported}")
    return MODEL_DIR / MODEL_FILENAMES[model_name]
