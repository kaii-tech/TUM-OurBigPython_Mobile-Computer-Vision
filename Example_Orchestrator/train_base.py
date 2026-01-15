#!/usr/bin/env python3
"""
VPR Distributed Training - Training Script for Individual Workstations

Run this on each Linux workstation:
    python train_base.py resnet50 ~/vpr_training/data
    python train_base.py efficientnetb0 ~/vpr_training/data
    python train_base.py vit ~/vpr_training/data
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load images from directory structure: data/class_name/*.jpg"""
    
    # Check if directories exist
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        sys.exit(1)
    
    print(f"Loading training data from {train_dir}...")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )
    
    print(f"Loading validation data from {val_dir}...")
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )
    
    return train_ds, val_ds

# ============================================================================
# Model Creation
# ============================================================================

def create_model(num_classes, base_model_name='resnet50'):
    """Create transfer learning model"""
    
    print(f"Creating {base_model_name} model with {num_classes} classes...")
    
    # Load base model
    if base_model_name == 'resnet50':
        base = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess_fn = keras.applications.resnet50.preprocess_input
        
    elif base_model_name == 'efficientnetb0':
        base = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess_fn = keras.applications.efficientnet.preprocess_input
        
    elif base_model_name == 'vit' or base_model_name == 'densenet121':
        base = keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess_fn = keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Freeze base model
    base.trainable = False
    
    # Build custom model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_fn(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base

# ============================================================================
# Training
# ============================================================================

def train_model(model, train_ds, val_ds, model_name, output_dir='./results'):
    """Train the model with checkpointing and early stopping"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    # Train
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*70}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(output_dir, f'{model_name}_final.keras')
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Save history
    history_path = os.path.join(output_dir, f'{model_name}_history.json')
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {max(history_dict['val_accuracy']):.4f}")
    print(f"{'='*70}\n")
    
    return history

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python train_base.py <model_name> [data_dir]")
        print("  model_name: resnet50, efficientnetb0, vit")
        print("  data_dir: path to data directory (default: ./data)")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    data_dir = sys.argv[2] if len(sys.argv) > 2 else './data'
    
    # Expand home directory
    data_dir = os.path.expanduser(data_dir)
    
    # Load data
    print(f"\n{'='*70}")
    print(f"VPR Transfer Learning Training")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*70}\n")
    
    train_ds, val_ds = load_data(data_dir)
    num_classes = len(train_ds.class_names)
    
    print(f"Found {num_classes} classes:")
    for i, class_name in enumerate(train_ds.class_names):
        print(f"  {i}: {class_name}")
    print()
    
    # Create model
    model, base = create_model(num_classes, model_name)
    print(f"\nBase model parameters: {base.count_params():,}")
    print(f"Total model parameters: {model.count_params():,}\n")
    
    # Train
    train_model(model, train_ds, val_ds, model_name)
