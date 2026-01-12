"""Simple CLI for running inference with a TensorFlow Lite room classifier.

Example:
    python tflite_inference.py \
        --model "Labcourse AI/room_quantized.tflite" \
        --labels "Labcourse AI/room_label_mapping.npy" \
        --image "sample.jpg"
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path



IMG_SIZE = (224, 224)
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "tflite_inference_paths.json"


def load_persisted_paths():
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return {
                key: Path(value)
                for key, value in data.items()
                if value
            }
    except Exception as exc:
        print(f"Warning: could not load saved paths ({exc})")
    return {}


PERSISTED_PATHS = load_persisted_paths()
DEFAULT_MODEL_CANDIDATES = [
    PERSISTED_PATHS.get("model"),
    SCRIPT_DIR / "room_quantized.tflite",
    SCRIPT_DIR / "room_quantized_fp16.tflite",
    SCRIPT_DIR / "room_model.tflite",
    SCRIPT_DIR / "room_model_dynamic.tflite",
]
DEFAULT_LABEL_CANDIDATES = [
    PERSISTED_PATHS.get("labels"),
    SCRIPT_DIR / "room_label_mapping.npy",
    SCRIPT_DIR.parent / "room_label_mapping.npy",
]


def save_persisted_paths(model_path, labels_path):
    payload = {
        "model": str(model_path),
        "labels": str(labels_path),
    }
    try:
        CONFIG_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"Warning: could not save paths ({exc})")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    SCRIPT_DIR / "room_quantized.tflite",
    SCRIPT_DIR / "room_quantized_fp16.tflite",
    SCRIPT_DIR / "room_model.tflite",
    SCRIPT_DIR / "room_model_dynamic.tflite",
]
DEFAULT_LABEL_CANDIDATES = [
    SCRIPT_DIR / "room_label_mapping.npy",
    SCRIPT_DIR.parent / "room_label_mapping.npy",
]


def load_label_map(path):
    mapping = np.load(path, allow_pickle=True).item()
    return {int(k): v for k, v in mapping.items()}


def pick_file(description, filetypes):
    """Let the user choose a file via dialog, fallback to console input."""
    try:
        from tkinter import Tk, filedialog
    except Exception:
        prompt = f"Enter path for {description}: "
        user_input = input(prompt).strip()
        return Path(user_input)

    root = Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(title=f"Select {description}", filetypes=filetypes)
    root.update()
    root.destroy()

    if not selected:
        raise ValueError(f"No {description.lower()} selected")
    return Path(selected)


def load_image(image_path):
    data = tf.io.read_file(image_path)
    image = tf.image.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def prepare_input(image, input_details):
    if input_details["dtype"] == np.uint8:
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            raise ValueError("Quantized model provided zero scale")
        image = image / scale + zero_point
        image = tf.clip_by_value(image, 0, 255)
        return tf.cast(image, tf.uint8)
    return tf.cast(image, tf.float32)


def decode_output(raw_output, output_details):
    tensor = tf.convert_to_tensor(raw_output)
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if scale == 0:
            raise ValueError("Output tensor reported zero quantization scale")
        tensor = tf.cast(tensor, tf.float32)
        tensor = (tensor - zero_point) * scale
    return tf.nn.softmax(tensor)


def run_inference(model_path, labels_path, image_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image = load_image(str(image_path))
    image = prepare_input(image, input_details)

    interpreter.set_tensor(input_details["index"], image.numpy())
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details["index"])[0]
    output = decode_output(raw_output, output_details).numpy()
    label_map = load_label_map(str(labels_path))

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    predicted_label = label_map[predicted_index]

    print(f"Prediction: {predicted_label} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a TFLite room classifier")
    parser.add_argument("--model", help="Path to .tflite model")
    parser.add_argument("--labels", help="Path to room_label_mapping.npy")
    parser.add_argument("--image", help="Path to image file")
    args = parser.parse_args()

    def resolve_path(value, description, filetypes, default_candidates=None):
        candidates = []
        if value:
            candidates.append(Path(value).expanduser())
        if default_candidates:
            for candidate in default_candidates:
                if candidate is None:
                    continue
                candidates.append(Path(candidate))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        path = pick_file(description, filetypes)
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path

    model_path = resolve_path(
        args.model,
        "TFLite model",
        [("TFLite model", "*.tflite"), ("All files", "*.*")],
        default_candidates=DEFAULT_MODEL_CANDIDATES,
    )
    labels_path = resolve_path(
        args.labels,
        "label map (room_label_mapping.npy)",
        [("NumPy file", "*.npy"), ("All files", "*.*")],
        default_candidates=DEFAULT_LABEL_CANDIDATES,
    )
    image_path = resolve_path(
        args.image,
        "input image",
        [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
    )

    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Image: {image_path}")

    save_persisted_paths(model_path, labels_path)
    run_inference(model_path, labels_path, image_path)
