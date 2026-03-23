import tensorflow as tf


SOURCE_MODEL = "my_model4.h5"
TARGET_MODEL = "my_model4.tflite"


def main():
    model = tf.keras.models.load_model(SOURCE_MODEL)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TARGET_MODEL, "wb") as f:
        f.write(tflite_model)

    print(f"Wrote {TARGET_MODEL} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()
