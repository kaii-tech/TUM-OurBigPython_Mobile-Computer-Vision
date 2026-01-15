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
import kagglehub

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


def load_and_preprocess_data():
    """Load and preprocess Lab Course Dataset."""
    print("\n" + "-" * 50)
    print("Loading Lab Course Dataset...")
    print("-" * 50)
    
    # Download latest version
    path = kagglehub.dataset_download("amaralibey/gsv-cities")
    print("Path to dataset files:", path)
    
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


def create_mobilenet_model():
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
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
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
    
    # Create model with MobileNetV2 base
    print("\n" + "-" * 50)
    print("Creating MobileNetV2 Model...")
    print("-" * 50)
    model, base_model = create_mobilenet_model()
    
    # ========== STAGE 1: Train classification head only ==========
    print("\n" + "=" * 60)
    print("STAGE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 60)
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR),
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
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=FINE_TUNE_LR),
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
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
