"""Check whether this computer is ready to run the leaflens.ai web application."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REQUIRED_PACKAGES = {
    "TensorFlow": "tensorflow",
    "NumPy": "numpy",
    "scikit-learn": "sklearn",
    "Matplotlib": "matplotlib",
    "Pillow": "PIL",
    "OpenCV": "cv2",
    "Flask": "flask",
}
REQUIRED_MODELS = ("crop_disease_mobilenetv2.keras",)
OPTIONAL_MODELS = ("crop_disease_efficientnetb0.keras",)


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "OK" if passed else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return passed


def main() -> int:
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    ready = check("Python 3.12", sys.version_info[:2] == (3, 12), "Install Python 3.12.x if this fails.")

    for label, module in REQUIRED_PACKAGES.items():
        ready &= check(label, importlib.util.find_spec(module) is not None, "Run: pip install -r requirements.txt")

    class_names = MODELS_DIR / "class_names.txt"
    class_count = 0
    if class_names.exists():
        class_count = len([line for line in class_names.read_text(encoding="utf-8").splitlines() if line])
    ready &= check("Class names", class_count > 0, "Copy models/class_names.txt into the models folder.")
    if class_count:
        print(f"       Found {class_count} plant classes.")

    for model_name in REQUIRED_MODELS:
        model_path = MODELS_DIR / model_name
        ready &= check(model_name, model_path.is_file() and model_path.stat().st_size > 1_000_000, "Copy this model file into the models folder.")

    for model_name in OPTIONAL_MODELS:
        model_path = MODELS_DIR / model_name
        check(model_name, model_path.is_file() and model_path.stat().st_size > 1_000_000, "Optional comparison model.")

    if ready:
        print("\nSetup is ready. Start the app with: python backend/app.py")
        return 0

    print("\nSetup is incomplete. Fix the MISSING items, then run this check again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
