from flask import Flask, request, jsonify
from PIL import Image
import io
import os
from model import get_model_status, start_model_warmup
from prediction import predict_image

app = Flask(__name__)
start_model_warmup()


@app.route("/health", methods=["GET"])
def health():
    model_status = get_model_status()
    return {"status": "ok", "model_status": model_status}


@app.route("/predict", methods=["POST"])
def predict():
    model_status = get_model_status()
    if model_status != "ready":
        return jsonify(
            {
                "error": "Model is still loading. Retry shortly.",
                "model_status": model_status,
            }
        ), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    result = predict_image(image)
    return jsonify(result)


if __name__ == "__main__":
    # For local development; Render will use gunicorn entrypoint instead
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
