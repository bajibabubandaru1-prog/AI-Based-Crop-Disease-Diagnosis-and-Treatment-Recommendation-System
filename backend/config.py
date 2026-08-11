"""Application configuration for the crop disease diagnosis web app."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    PROJECT_ROOT = PROJECT_ROOT
    ASSET_VERSION = "2026.08.11.1"
    SECRET_KEY = "development-only-change-this-before-deployment"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    CONFIDENCE_THRESHOLD = 0.60
    TOP_K_PREDICTIONS = 3
    REMOTE_IMAGE_TIMEOUT_SECONDS = 12
