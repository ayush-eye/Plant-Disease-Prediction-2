from pathlib import Path
import threading

MODEL_PATH = Path(__file__).with_name("my_model4.tflite")

_interpreter = None
_input_details = None
_output_details = None
_load_error = None
_load_lock = threading.Lock()
_warmup_started = False

CLASS_MAPPING = {
    0: "Apple scab",
    1: "Apple Black rot",
    2: "Cedar Apple rust",
    3: "Apple healthy",
    4: "Blueberry healthy",
    5: "Cherry Powdery mildew",
    6: "Cherry healthy",
    7: "Corn (maize) Cercospora leaf spot Gray leaf spot",
    8: "Corn (maize) Common rust",
    9: "Corn (maize) Northern Leaf Blight",
    10: "Corn (maize) healthy",
    11: "Grape Black rot",
    12: "Grape Esca (Black Measles)",
    13: "Grape Leaf blight (Isariopsis Leaf Spot)",
    14: "Grape healthy",
    15: "Orange Haunglongbing (Citrus greening)",
    16: "Peach Bacterial spot",
    17: "Peach healthy",
    18: "Pepper, bell Bacterial spot",
    19: "Pepper, bell healthy",
    20: "Potato Early blight",
    21: "Potato Late blight",
    22: "Potato healthy",
    23: "Raspberry healthy",
    24: "Soybean healthy",
    25: "Squash Powdery mildew",
    26: "Strawberry Leaf scorch",
    27: "Strawberry healthy",
    28: "Tomato Bacterial spot",
    29: "Tomato Early blight",
    30: "Tomato Late blight",
    31: "Tomato Leaf Mold",
    32: "Tomato Septoria leaf spot",
    33: "Tomato Spider mites Two spotted spider mite",
    34: "Tomato Target Spot",
    35: "Tomato Tomato Yellow Leaf Curl Virus",
    36: "Tomato Tomato mosaic virus",
    37: "Tomato healthy"
}


def _load_model_once():
    global _interpreter, _input_details, _output_details, _load_error

    if _interpreter is not None:
        return _interpreter, _input_details, _output_details

    if _load_error is not None:
        raise _load_error

    with _load_lock:
        if _interpreter is not None:
            return _interpreter, _input_details, _output_details
        if _load_error is not None:
            raise _load_error

        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter

            _interpreter = Interpreter(model_path=str(MODEL_PATH))
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()[0]
            _output_details = _interpreter.get_output_details()[0]
            return _interpreter, _input_details, _output_details
        except Exception as exc:
            _load_error = exc
            raise


def get_model():
    return _load_model_once()


def is_model_ready():
    return _interpreter is not None


def get_model_status():
    if _interpreter is not None:
        return "ready"
    if _load_error is not None:
        return "error"
    if _warmup_started:
        return "loading"
    return "idle"


def start_model_warmup():
    global _warmup_started

    if _warmup_started or _interpreter is not None:
        return

    _warmup_started = True

    def _warm():
        try:
            _load_model_once()
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True, name="model-warmup").start()
