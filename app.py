from flask import Flask, request, jsonify
from PIL import Image
import io
import os
from prediction import predict_image

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
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
