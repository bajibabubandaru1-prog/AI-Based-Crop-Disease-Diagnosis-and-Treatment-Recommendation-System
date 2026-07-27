import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from model_service import DEFAULT_MODEL_NAME, MODEL_FILENAMES, predict_leaf_image
from recommendations import get_recommendation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
CONFIDENCE_THRESHOLD = 0.60


def create_app():
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            models=MODEL_FILENAMES.keys(),
            default_model=DEFAULT_MODEL_NAME,
        )

    @app.post("/api/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No image file was uploaded."}), 400

        upload = request.files["image"]
        if not upload.filename:
            return jsonify({"error": "Please select an image file."}), 400

        extension = upload.filename.rsplit(".", 1)[-1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Only JPG, JPEG, and PNG files are supported."}), 400

        model_name = request.form.get("model", DEFAULT_MODEL_NAME)
        filename = secure_filename(upload.filename)
        saved_name = f"{uuid.uuid4().hex}_{filename}"
        image_path = UPLOAD_DIR / saved_name
        upload.save(image_path)

        try:
            result = predict_leaf_image(image_path, model_name=model_name)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        top_prediction = result["top_prediction"]
        confidence = top_prediction["confidence"]
        class_name = top_prediction["class_name"]
        result["recommendation"] = get_recommendation(class_name)
        result["image_url"] = f"/static/uploads/{saved_name}"
        result["is_confident"] = confidence >= CONFIDENCE_THRESHOLD
        result["confidence_threshold"] = CONFIDENCE_THRESHOLD

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
