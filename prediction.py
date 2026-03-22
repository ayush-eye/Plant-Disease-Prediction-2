import numpy as np
from tensorflow.keras.preprocessing import image
from model import model, CLASS_MAPPING

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    return {
        "disease": CLASS_MAPPING.get(class_index, "Unknown"),
        "confidence": confidence,
        "model_version": "v1.0.0"
    }
