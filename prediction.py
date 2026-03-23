import numpy as np
from model import CLASS_MAPPING, get_model


def predict_image(img):
    img = img.resize((224, 224))
    interpreter, input_details, output_details = get_model()

    img_array = np.asarray(img, dtype=input_details["dtype"])
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details["index"])
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    return {
        "disease": CLASS_MAPPING.get(class_index, "Unknown"),
        "confidence": confidence,
        "model_version": "v1.0.0"
    }
