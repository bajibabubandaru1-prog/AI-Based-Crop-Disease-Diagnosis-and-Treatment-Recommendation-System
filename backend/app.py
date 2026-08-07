"""Flask application for crop disease image diagnosis."""

import ipaddress
import socket
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

try:
    from .config import Config
    from .model_service import (
        DEFAULT_MODEL_NAME,
        get_available_models,
        load_class_names,
        predict_leaf_image,
    )
    from .recommendations import get_recommendation
except ImportError:  # Allows: python backend/app.py
    from config import Config
    from model_service import DEFAULT_MODEL_NAME, get_available_models, load_class_names, predict_leaf_image
    from recommendations import get_recommendation


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder=str(config_class.PROJECT_ROOT / "templates"),
        static_folder=str(config_class.PROJECT_ROOT / "static"),
    )
    app.config.from_object(config_class)
    app.config["UPLOAD_DIR"] = config_class.UPLOAD_DIR
    config_class.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.after_request
    def disable_cache_during_development(response):
        """Ensure phones receive the latest template, styles, and script after edits."""
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    def get_available_model_ids():
        return {model["id"] for model in get_available_models() if model["available"]}

    def get_requested_model():
        model_name = request.form.get("model", DEFAULT_MODEL_NAME)
        if model_name not in get_available_model_ids():
            raise ValueError("The selected model is not available. Choose a trained model.")
        return model_name

    def build_prediction_response(image_path: Path, saved_name: str, model_name: str):
        result = predict_leaf_image(
            image_path,
            model_name=model_name,
            top_k=app.config["TOP_K_PREDICTIONS"],
        )
        top_prediction = result["top_prediction"]
        is_accepted = top_prediction["confidence"] >= app.config["CONFIDENCE_THRESHOLD"]
        if is_accepted:
            recommendation = get_recommendation(top_prediction["class_name"])
        else:
            recommendation = {
                "display_name": "No reliable leaf diagnosis",
                "organic_treatment": "No treatment recommendation is provided because the image was not confidently recognised as a supported crop condition.",
                "chemical_treatment": "Do not apply treatment based on this result. Submit a clear image of one leaf or seek agricultural advice.",
                "prevention": "Use a close, well-lit image with a single leaf filling most of the frame. Non-leaf images are outside this model's scope.",
            }
        result.update({
            "recommendation": recommendation,
            "image_url": f"/static/uploads/{saved_name}",
            "is_confident": is_accepted,
            "diagnosis_accepted": is_accepted,
            "confidence_threshold": app.config["CONFIDENCE_THRESHOLD"],
        })
        return result

    def validate_public_image_url(value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Paste a direct public image link starting with http:// or https://.")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
            resolved_ips = {item[4][0] for item in addresses}
        except socket.gaierror as exc:
            raise ValueError("That image link could not be reached.") from exc
        for address in resolved_ips:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError("Only public image links can be used.")
        return parsed.geturl()

    def download_remote_image(image_url: str, destination: Path) -> None:
        request_headers = {"User-Agent": "LeafLens/1.0 image diagnosis"}
        try:
            with urlopen(
                Request(image_url, headers=request_headers),
                timeout=app.config["REMOTE_IMAGE_TIMEOUT_SECONDS"],
            ) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise ValueError("That link does not point to an image file.")
                bytes_written = 0
                with destination.open("wb") as output:
                    while chunk := response.read(64 * 1024):
                        bytes_written += len(chunk)
                        if bytes_written > app.config["MAX_CONTENT_LENGTH"]:
                            raise ValueError("The linked image is larger than 10 MB.")
                        output.write(chunk)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ValueError("The image link could not be downloaded. Try a direct public image URL.") from exc

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            models=get_available_models(),
            default_model=DEFAULT_MODEL_NAME,
            confidence_threshold=app.config["CONFIDENCE_THRESHOLD"],
            asset_version=app.config["ASSET_VERSION"],
        )

    @app.get("/showcase-leaf")
    def showcase_leaf():
        sample_path = config_class.PROJECT_ROOT / "dataset" / "external_samples" / "tomato_late_blight_1.jpg"
        if not sample_path.exists():
            return "", 404
        return send_file(sample_path, mimetype="image/jpeg", max_age=3600)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "models": get_available_models(),
            "class_count": len(load_class_names()),
        })

    @app.get("/api/classes")
    def classes():
        return jsonify({"classes": load_class_names()})

    @app.post("/api/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "Select a JPG or PNG leaf image before predicting."}), 400

        upload = request.files["image"]
        if not upload.filename:
            return jsonify({"error": "Select an image file before predicting."}), 400

        filename = secure_filename(upload.filename)
        if "." not in filename:
            return jsonify({"error": "Use a JPG, JPEG, or PNG image."}), 400
        extension = filename.rsplit(".", 1)[1].lower()
        if extension not in app.config["ALLOWED_EXTENSIONS"]:
            return jsonify({"error": "Only JPG, JPEG, and PNG image files are supported."}), 400

        try:
            model_name = get_requested_model()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        saved_name = f"{uuid.uuid4().hex}.{extension}"
        image_path = Path(app.config["UPLOAD_DIR"]) / saved_name
        try:
            upload.save(image_path)
        except OSError:
            app.logger.exception("Could not save uploaded image")
            return jsonify({"error": "The uploaded image could not be saved. Please try again."}), 500

        try:
            result = build_prediction_response(image_path, saved_name, model_name)
        except ValueError as exc:
            image_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except (FileNotFoundError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify({"error": "Prediction could not be completed. Please try another image."}), 500

        return jsonify(result)

    @app.post("/api/predict-url")
    def predict_from_url():
        image_url = request.form.get("image_url", "")
        if not image_url.strip():
            return jsonify({"error": "Paste a direct public image URL before analyzing."}), 400
        try:
            model_name = get_requested_model()
            safe_url = validate_public_image_url(image_url)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        saved_name = f"{uuid.uuid4().hex}.jpg"
        image_path = Path(app.config["UPLOAD_DIR"]) / saved_name
        try:
            download_remote_image(safe_url, image_path)
            result = build_prediction_response(image_path, saved_name, model_name)
        except ValueError as exc:
            image_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except (FileNotFoundError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception:
            image_path.unlink(missing_ok=True)
            app.logger.exception("URL prediction failed")
            return jsonify({"error": "The linked image could not be analyzed. Try another direct image URL."}), 500
        return jsonify(result)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return jsonify({"error": "Image is too large. Choose an image smaller than 10 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
