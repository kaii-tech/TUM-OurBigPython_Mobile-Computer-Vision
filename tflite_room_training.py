#!/usr/bin/env python3
"""
Room Classification Training Script
This script trains a CNN model on the Lab Course Dataset and converts it to TFLite.
Run via WSL for GPU acceleration: .\run_wsl.ps1 "tflite_room_training.py"
"""

import tensorflow as tf
import numpy as np
import os
import time
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

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
INIT_LR = 1e-3
N_EPOCH = 10
BATCH_SIZE = 32
IMG_SIZE = 224  # Image size for room classification
NUM_CLASSES = 27  # 27 rooms in the dataset

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


def load_and_preprocess_data():
    """Load and preprocess Lab Course Dataset."""
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
    
    images = []
    labels = []
    
    # Load images from each folder
    for room_id in range(1, 28):  # 1 to 27
        folder_path = dataset_path / str(room_id)
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} not found")
            continue
        
        room_name = id_to_room[room_id]
        label = room_to_label[room_name]
        
        # Count images in this folder
        image_files = list(folder_path.glob('*.[jJ][pP][gG]')) + \
                      list(folder_path.glob('*.[jJ][pP][eE][gG]')) + \
                      list(folder_path.glob('*.[pP][nN][gG]'))
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"  Room {room_id} ({room_name}): {len(image_files)} images")
    
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


def create_cnn_model():
    """Create a CNN model for room classification (27 classes)."""
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        tf.keras.Input(shape=input_shape),
        data_augmentation,

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
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
    
    # Create model
    print("\n" + "-" * 50)
    print("Creating Model...")
    print("-" * 50)
    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n" + "-" * 50)
    print("Starting Training...")
    print(f"Epochs: {N_EPOCH}, Batch size: {BATCH_SIZE}")
    print("-" * 50)
    
    history = model.fit(
        x_train, y_train,
        epochs=N_EPOCH,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "-" * 50)
    print("Evaluating Model...")
    print("-" * 50)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
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
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
