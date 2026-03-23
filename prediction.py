import numpy as np
from model import CLASS_MAPPING, get_model

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    model = get_model()
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    return {
        "disease": CLASS_MAPPING.get(class_index, "Unknown"),
        "confidence": confidence,
        "model_version": "v1.0.0"
    }
