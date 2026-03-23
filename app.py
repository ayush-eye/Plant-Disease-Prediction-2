from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import io
import os
from model import get_model_error_message, get_model_status
from prediction import predict_image

app = Flask(__name__)


def _build_liveness_response():
    return {
        "status": "ok",
        "service": "plant-disease-api",
        "model_status": get_model_status(),
    }


def _build_health_response():
    model_status = get_model_status()
    response = {"model_status": model_status}

    if model_status == "ready":
        response["status"] = "ok"
        return response, 200

    if model_status == "error":
        response["status"] = "error"
        model_error = get_model_error_message()
        if model_error:
            response["model_error"] = model_error
        return response, 503

    response["status"] = "starting"
    return response, 200


@app.route("/", methods=["GET"])
def index():
    return jsonify(_build_liveness_response()), 200


@app.route("/health", methods=["GET"])
def health():
    response, status_code = _build_health_response()
    return jsonify(response), status_code


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Image filename is empty"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Image file is empty"}), 400

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        result = predict_image(image)
    except Exception:
        model_status = get_model_status()
        if model_status != "ready":
            response = {
                "error": "Model unavailable",
                "model_status": model_status,
            }
            model_error = get_model_error_message()
            if model_error:
                response["model_error"] = model_error
            return jsonify(response), 503
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify(result)


if __name__ == "__main__":
    # For local development; Render will use gunicorn entrypoint instead
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
