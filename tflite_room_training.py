#!/usr/bin/env python3
"""
Room Classification Training Script
This script trains a CNN model on the Lab Course Dataset and converts it to TFLite.
Run via WSL for GPU acceleration: .\run_wsl.ps1 "tflite_room_training.py"
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_AUTOGRAPH_VERBOSITY'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

import tensorflow as tf
import numpy as np
import time
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import seaborn as sns

# Configure TensorFlow to use GPU in WSL
print("=" * 60)
print("TensorFlow Room Classification Training")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU.")

# Hyperparameters
INIT_LR = 2e-3  # Increased for batch size 64
FINE_TUNE_LR = 1e-4  # Lower LR for fine-tuning
N_EPOCH_HEAD = 30  # Epochs for training head only
N_EPOCH_FINETUNE = 50  # Epochs for fine-tuning
BATCH_SIZE = 64  # Increased from 32 for faster convergence
IMG_SIZE = 224  # Image size for room classification
NUM_CLASSES = 27  # 27 rooms in the dataset
FINE_TUNE_LAYERS = 100  # Number of layers to unfreeze for fine-tuning

# Set random seeds
tf.random.set_seed(251211)
np.random.seed(251211)

# Get script directory for WSL-compatible paths
script_dir = Path(__file__).parent.resolve()
dataset_path = script_dir / "Dataset"
index_path = dataset_path / "index.json"

print(f"\nScript directory: {script_dir}")
print(f"Dataset path: {dataset_path}")


def load_room_labels():
    """Load room labels from index.json."""
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    # Create mapping from id to room name
    id_to_room = {}
    for entry in index_data:
        room_id = entry['id']
        room_name = entry['room']
        id_to_room[room_id] = room_name
    
    return id_to_room, index_data


def load_single_image(img_path, label, img_size):
    """Load and preprocess a single image (thread-safe)."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array, label, None
    except Exception as e:
        return None, None, f"Error loading {img_path}: {e}"


def load_and_preprocess_data():
    """Load and preprocess Lab Course Dataset with parallel image loading."""
    print("\n" + "-" * 50)
    print("Loading Lab Course Dataset...")
    print("-" * 50)
    
    id_to_room, index_data = load_room_labels()
    print(f"Found {len(id_to_room)} room classes")
    
    # Create label mapping (room name to integer)
    room_names = sorted(set(id_to_room.values()))
    room_to_label = {name: idx for idx, name in enumerate(room_names)}
    label_to_room = {idx: name for name, idx in room_to_label.items()}
    
    print(f"Room classes: {room_names}")
    
    # Collect all image paths and labels first
    image_tasks = []
    room_counts = defaultdict(int)
    
    for room_id in range(1, 28):  # 1 to 27
        folder_path = dataset_path / str(room_id)
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} not found")
            continue
        
        room_name = id_to_room[room_id]
        label = room_to_label[room_name]
        
        # Collect image files
        image_files = list(folder_path.glob('*.[jJ][pP][gG]')) + \
                      list(folder_path.glob('*.[jJ][pP][eE][gG]')) + \
                      list(folder_path.glob('*.[pP][nN][gG]'))
        
        for img_path in image_files:
            image_tasks.append((img_path, label, room_id, room_name))
        
        room_counts[room_id] = (len(image_files), room_name)
    
    total_images = len(image_tasks)
    print(f"\nTotal images to load: {total_images}")
    print("Loading images in parallel...")
    
    # Parallel image loading
    images = []
    labels = []
    errors = []
    
    num_workers = min(32, (os.cpu_count() or 4) * 4)  # Optimal for I/O-bound tasks
    start_time = time.time()
    completed = 0
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(load_single_image, img_path, label, IMG_SIZE): (img_path, room_id, room_name)
            for img_path, label, room_id, room_name in image_tasks
        }
        
        for future in as_completed(future_to_task):
            img_array, label, error = future.result()
            
            with lock:
                completed += 1
                if completed % 500 == 0 or completed == total_images:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Loaded {completed}/{total_images} images ({rate:.1f} img/s)")
            
            if error:
                errors.append(error)
            elif img_array is not None:
                images.append(img_array)
                labels.append(label)
    
    elapsed_time = time.time() - start_time
    print(f"\nLoaded {len(images)} images in {elapsed_time:.2f}s ({len(images)/elapsed_time:.1f} img/s)")
    
    # Print room statistics
    print("\nImages per room:")
    for room_id in sorted(room_counts.keys()):
        count, room_name = room_counts[room_id]
        print(f"  Room {room_id} ({room_name}): {count} images")
    
    if errors:
        print(f"\n{len(errors)} errors occurred during loading")
        for err in errors[:5]:
            print(f"  {err}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train[0].shape}")
    
    # Save label mapping for later use
    np.save(script_dir / 'label_mapping.npy', label_to_room)
    
    return (x_train, y_train), (x_test, y_test), label_to_room


def create_mobilenet_model(num_classes):
    """Create a MobileNetV2 transfer learning model for room classification."""
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name='data_augmentation')
    
    # Load MobileNetV2 pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    # Preprocess input for MobileNetV2 (scales to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model


def unfreeze_model(model, base_model, num_layers_to_unfreeze=FINE_TUNE_LAYERS):
    """Unfreeze the top layers of the base model for fine-tuning."""
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfroze {trainable_count} layers for fine-tuning")
    
    return model


def convert_to_tflite_float16(model, output_path):
    """Convert model to TFLite with float16 quantization."""
    print("\n" + "-" * 50)
    print("Converting to TFLite (Float16 Quantization)...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")

    return output_path


def main():
    # Load data
    train_data, test_data, label_to_room = load_and_preprocess_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print(f"\nLabel mapping: {label_to_room}")
    
    # Get actual number of classes
    num_classes = len(label_to_room)
    print(f"Number of classes: {num_classes}")
    
    # Create model with MobileNetV2 base
    print("\n" + "-" * 50)
    print("Creating MobileNetV2 Model...")
    print("-" * 50)
    model, base_model = create_mobilenet_model(num_classes)
    
    # ========== STAGE 1: Train classification head only ==========
    print("\n" + "=" * 60)
    print("STAGE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 60)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks for Stage 1
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    print(f"\nStage 1: {N_EPOCH_HEAD} epochs, Batch size: {BATCH_SIZE}, LR: {INIT_LR}")
    
    history1 = model.fit(
        x_train, y_train,
        epochs=N_EPOCH_HEAD,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    # Evaluate after Stage 1
    stage1_loss, stage1_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nStage 1 Results - Loss: {stage1_loss:.4f}, Accuracy: {stage1_acc:.4f} ({stage1_acc*100:.2f}%)")
    
    # ========== STAGE 2: Fine-tune top layers of base model ==========
    print("\n" + "=" * 60)
    print(f"STAGE 2: Fine-tuning (Unfreezing top {FINE_TUNE_LAYERS} layers)")
    print("=" * 60)
    
    # Unfreeze top layers
    model = unfreeze_model(model, base_model, FINE_TUNE_LAYERS)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for Stage 2 (more patience for fine-tuning)
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    print(f"\nStage 2: {N_EPOCH_FINETUNE} epochs, Batch size: {BATCH_SIZE}, LR: {FINE_TUNE_LR}")
    
    history2 = model.fit(
        x_train, y_train,
        epochs=N_EPOCH_FINETUNE,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\nImprovement from Stage 1: {(test_accuracy - stage1_acc)*100:.2f}%")
    
    # Save model
    model_path = script_dir / 'room_model.keras'
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save test data
    np.savez(script_dir / 'test_data.npz', x_test=x_test, y_test=y_test)
    print("Test data saved to: test_data.npz")
    
    # Convert to TFLite
    tflite_path = script_dir / 'room_model_float16.tflite'
    convert_to_tflite_float16(model, str(tflite_path))
    
    # Print size comparison
    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)
    keras_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    compression = keras_size / tflite_size
    print(f"Original Keras: {keras_size:.2f} MB")
    print(f"TFLite Float16: {tflite_size:.2f} MB (compression: {compression:.2f}x)")
    
    # ========== TEST TFLITE MODEL ==========
    print("\n" + "=" * 60)
    print("TESTING TFLITE MODEL ON TEST SET")
    print("=" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    
    # Run inference on test set
    tflite_predictions = []
    start_time = time.time()
    
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        tflite_predictions.append(np.argmax(output_data))
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(x_test)} images...")
    
    tflite_time = time.time() - start_time
    tflite_predictions = np.array(tflite_predictions)
    
    # Calculate TFLite accuracy
    tflite_accuracy = np.mean(tflite_predictions == y_test)
    print(f"\nTFLite Model Results:")
    print(f"  Accuracy: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
    print(f"  Inference time: {tflite_time:.2f}s ({len(x_test)/tflite_time:.1f} img/s)")
    print(f"  Accuracy difference from Keras: {(tflite_accuracy - test_accuracy)*100:+.2f}%")
    
    # ========== CONFUSION MATRIX & PER-ROOM ACCURACY ==========
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX & PER-ROOM ACCURACY (TFLite Model)")
    print("=" * 60)
    
    # Confusion matrix from TFLite predictions
    num_classes_actual = len(label_to_room)
    cm = confusion_matrix(y_test, tflite_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    room_labels = [label_to_room[i] for i in range(num_classes_actual)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=room_labels, yticklabels=room_labels)
    plt.title('Confusion Matrix - Room Classification (TFLite Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = script_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # Per-room accuracy
    print("\nTFLite Per-Room Accuracy:")
    print("-" * 40)
    for i in range(num_classes_actual):
        room_name = label_to_room[i]
        room_total = np.sum(y_test == i)
        room_correct = cm[i, i]
        room_acc = (room_correct / room_total * 100) if room_total > 0 else 0
        print(f"  {room_name:20s}: {room_acc:6.2f}% ({room_correct}/{room_total})")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
