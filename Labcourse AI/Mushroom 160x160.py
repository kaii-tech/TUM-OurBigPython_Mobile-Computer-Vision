"""
Lightweight CNN for Mushroom Classification (Mobile Target)
=============================================================
A complete pipeline for training, pruning, and quantizing a CNN for mobile deployment.

Tech Stack: Python, TensorFlow/Keras, NumPy, Pandas
Target: TFLite model optimized for mobile devices
"""

import os
# Set environment variable to use tf_keras with TFMOT before importing TensorFlow
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa  # For advanced image augmentations

# Use tf_keras for compatibility with tensorflow_model_optimization
import tf_keras as keras
from tf_keras import layers, models, callbacks, optimizers
import tensorflow_model_optimization as tfmot

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Data paths
    "data_dir": r"C:\Users\Klein\Downloads\archive",
    "image_base_dir": r"C:\Users\Klein\Downloads\archive\merged_dataset",
    
    # Image settings
    "img_height": 224,
    "img_width": 224,
    "channels": 3,
    
    # Training settings
    "batch_size": 32,
    "epochs_initial": 20,
    "epochs_pruning": 10,
    "learning_rate": 0.001,
    
    # Pruning settings
    "pruning_initial_sparsity": 0.0,
    "pruning_final_sparsity": 0.5,
    
    # Output paths
    "model_save_path": r"C:\Users\Klein\Desktop\Labcourse\Labcourse AI\mushroom_model.keras",
    "pruned_model_path": r"C:\Users\Klein\Desktop\Labcourse\Labcourse AI\mushroom_pruned.keras",
    "tflite_model_path": r"C:\Users\Klein\Desktop\Labcourse\Labcourse AI\mushroom_model.tflite",
    "tflite_quantized_path": r"C:\Users\Klein\Desktop\Labcourse\Labcourse AI\mushroom_quantized.tflite",
}

# =============================================================================
# Data Pipeline
# =============================================================================
def load_and_preprocess_data():
    """Load CSV files and create label mappings."""
    print("=" * 60)
    print("Loading dataset from CSV files...")
    print("=" * 60)
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "val.csv"))
    test_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "test.csv"))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Fix image paths (convert Kaggle paths to local paths)
    def fix_path(kaggle_path):
        # Extract relative path from Kaggle format
        # /kaggle/working/merged_dataset/ClassName/image.jpg -> ClassName/image.jpg
        parts = kaggle_path.replace("/kaggle/working/merged_dataset/", "")
        return os.path.join(CONFIG["image_base_dir"], parts)
    
    train_df["local_path"] = train_df["image_path"].apply(fix_path)
    val_df["local_path"] = val_df["image_path"].apply(fix_path)
    test_df["local_path"] = test_df["image_path"].apply(fix_path)
    
    # Create label encoding
    all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique()
    all_labels = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    num_classes = len(all_labels)
    print(f"Number of classes: {num_classes}")
    
    # Encode labels
    train_df["label_idx"] = train_df["label"].map(label_to_idx)
    val_df["label_idx"] = val_df["label"].map(label_to_idx)
    test_df["label_idx"] = test_df["label"].map(label_to_idx)
    
    return train_df, val_df, test_df, label_to_idx, idx_to_label, num_classes


def create_tf_dataset(df, is_training=False):
    """Create a tf.data.Dataset from a DataFrame with efficient prefetching."""
    
    image_paths = df["local_path"].values
    labels = df["label_idx"].values
    
    def load_and_preprocess_image(path, label):
        """Load image from path and preprocess."""
        # Read file
        img = tf.io.read_file(path)
        # Decode image (handles both jpg and jpeg)
        img = tf.image.decode_jpeg(img, channels=CONFIG["channels"])
        # Resize
        img = tf.image.resize(img, [CONFIG["img_height"], CONFIG["img_width"]])
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    def augment(image, label):
        """Apply domain-specific data augmentation for mushroom images."""
        # Horizontal flip is safe - mushrooms look similar from left/right
        image = tf.image.random_flip_left_right(image)
        
        # NO vertical flip - mushrooms have distinct up (cap) and down (stem)
        # NO 90-degree rotations - side profiles shouldn't appear sideways
        
        # Small rotation (±20 degrees) to simulate tilted camera
        # Using random affine-like transform via padding and cropping
        angle = tf.random.uniform([], minval=-0.35, maxval=0.35)  # ~±20 degrees in radians
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR', fill_mode='reflect')
        
        # Random zoom/crop - critical for mushrooms as users rarely frame perfectly
        # Randomly crop between 80-100% of the image, then resize back
        crop_fraction = tf.random.uniform([], minval=0.8, maxval=1.0)
        crop_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * crop_fraction, tf.int32)
        image = tf.image.random_crop(image, [crop_size[0], crop_size[1], CONFIG["channels"]])
        image = tf.image.resize(image, [CONFIG["img_height"], CONFIG["img_width"]])
        
        # Color augmentations - lighting varies in forest environments
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)  # Slight hue shift for varying light
        
        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    
    # Load and preprocess images (parallel)
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if training
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(CONFIG["batch_size"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# =============================================================================
# Model Architecture
# =============================================================================
def build_lightweight_cnn(num_classes):
    """
    Build a lightweight CNN optimized for mobile deployment.
    
    Architecture designed for:
    - Small model size
    - Fast inference
    - Good accuracy on mushroom classification
    """
    print("\n" + "=" * 60)
    print("Building Lightweight CNN Architecture...")
    print("=" * 60)
    
    model = models.Sequential([
        # Input layer
        layers.InputLayer(input_shape=(CONFIG["img_height"], CONFIG["img_width"], CONFIG["channels"])),
        
        # Block 1: Initial feature extraction
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: Deeper features
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3: More complex patterns
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4: High-level features
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 5: Final conv block with depthwise separable for efficiency
        layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu', name='sep_conv5'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # More efficient than Flatten
        
        # Classification head
        layers.Dense(256, activation='relu', name='dense1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu', name='dense2'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.summary()
    return model


def compile_model(model):
    """Compile the model with optimizer and loss function."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================================================
# Training
# =============================================================================
def train_model(model, train_dataset, val_dataset):
    """Train the model with callbacks."""
    print("\n" + "=" * 60)
    print("Training Model...")
    print("=" * 60)
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG["epochs_initial"],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history


# =============================================================================
# Pruning (TFMOT)
# =============================================================================
def apply_pruning(model, train_dataset, val_dataset):
    """Apply magnitude-based pruning using TensorFlow Model Optimization Toolkit."""
    print("\n" + "=" * 60)
    print("Applying Magnitude-Based Pruning...")
    print("=" * 60)
    
    # Calculate pruning schedule
    num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    end_step = num_batches * CONFIG["epochs_pruning"]
    
    # Pruning configuration
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=CONFIG["pruning_initial_sparsity"],
            final_sparsity=CONFIG["pruning_final_sparsity"],
            begin_step=0,
            end_step=end_step
        )
    }
    
    # Apply pruning to the model
    # Clone the model and apply pruning wrapper
    def apply_pruning_to_layer(layer):
        """Apply pruning to Conv2D and Dense layers."""
        if isinstance(layer, (layers.Conv2D, layers.Dense, layers.SeparableConv2D)):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer
    
    # Clone and apply pruning
    pruned_model = keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_layer
    )
    
    # Compile pruned model
    pruned_model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"] * 0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Pruning callbacks
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs'),
    ]
    
    print(f"Fine-tuning with pruning for {CONFIG['epochs_pruning']} epochs...")
    
    # Fine-tune with pruning
    pruned_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG["epochs_pruning"],
        callbacks=pruning_callbacks,
        verbose=1
    )
    
    # Strip pruning wrappers for export
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # Print model size comparison
    print("\nPruning Results:")
    print(f"Original model parameters: {model.count_params():,}")
    print(f"Pruned model parameters: {final_model.count_params():,}")
    
    return final_model


# =============================================================================
# Quantization & TFLite Conversion
# =============================================================================
def convert_to_tflite(model, val_dataset):
    """Convert model to TFLite with various quantization options."""
    print("\n" + "=" * 60)
    print("Converting to TFLite Format...")
    print("=" * 60)
    
    # Create representative dataset for full integer quantization
    def representative_dataset_gen():
        """Generator for representative dataset used in quantization."""
        for images, _ in val_dataset.take(100):
            for i in range(images.shape[0]):
                yield [tf.expand_dims(images[i], axis=0)]
    
    # 1. Dynamic Range Quantization (smallest, fastest)
    print("\n1. Creating Dynamic Range Quantized model...")
    converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter_dynamic.convert()
    
    dynamic_path = CONFIG["tflite_model_path"].replace(".tflite", "_dynamic.tflite")
    with open(dynamic_path, 'wb') as f:
        f.write(tflite_dynamic)
    print(f"   Saved to: {dynamic_path}")
    print(f"   Size: {len(tflite_dynamic) / 1024 / 1024:.2f} MB")
    
    # 2. Full Integer Quantization (best for mobile)
    print("\n2. Creating Full Integer Quantized model...")
    converter_int = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int.representative_dataset = representative_dataset_gen
    converter_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int.inference_input_type = tf.uint8
    converter_int.inference_output_type = tf.uint8
    
    try:
        tflite_int = converter_int.convert()
        int_path = CONFIG["tflite_quantized_path"]
        with open(int_path, 'wb') as f:
            f.write(tflite_int)
        print(f"   Saved to: {int_path}")
        print(f"   Size: {len(tflite_int) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   Full integer quantization failed: {e}")
        print("   Falling back to float16 quantization...")
        
        # Fallback: Float16 quantization
        converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_fp16.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter_fp16.convert()
        
        fp16_path = CONFIG["tflite_quantized_path"].replace(".tflite", "_fp16.tflite")
        with open(fp16_path, 'wb') as f:
            f.write(tflite_fp16)
        print(f"   Saved to: {fp16_path}")
        print(f"   Size: {len(tflite_fp16) / 1024 / 1024:.2f} MB")
    
    # 3. Standard TFLite (no quantization, for comparison)
    print("\n3. Creating Standard TFLite model (no quantization)...")
    converter_std = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_std = converter_std.convert()
    
    std_path = CONFIG["tflite_model_path"]
    with open(std_path, 'wb') as f:
        f.write(tflite_std)
    print(f"   Saved to: {std_path}")
    print(f"   Size: {len(tflite_std) / 1024 / 1024:.2f} MB")
    
    return tflite_dynamic


def evaluate_tflite_model(tflite_model_path, test_dataset):
    """Evaluate TFLite model accuracy."""
    print(f"\nEvaluating TFLite model: {tflite_model_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    
    for images, labels in test_dataset:
        for i in range(images.shape[0]):
            # Prepare input
            input_data = tf.expand_dims(images[i], axis=0)
            
            # Handle quantized input
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = input_data / input_scale + input_zero_point
                input_data = tf.cast(input_data, tf.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted = np.argmax(output_data)
            
            if predicted == labels[i].numpy():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"TFLite Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """Execute the complete training and optimization pipeline."""
    print("\n" + "=" * 60)
    print("MUSHROOM CLASSIFICATION - MOBILE OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Set memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected, using CPU")
    
    # Step 1: Load and preprocess data
    train_df, val_df, test_df, label_to_idx, idx_to_label, num_classes = load_and_preprocess_data()
    
    # Save label mappings for inference
    label_mapping_path = os.path.join(
        os.path.dirname(CONFIG["model_save_path"]), 
        "label_mapping.npy"
    )
    np.save(label_mapping_path, idx_to_label)
    print(f"Label mapping saved to: {label_mapping_path}")
    
    # Step 2: Create tf.data datasets
    print("\nCreating tf.data pipelines...")
    train_dataset = create_tf_dataset(train_df, is_training=True)
    val_dataset = create_tf_dataset(val_df, is_training=False)
    test_dataset = create_tf_dataset(test_df, is_training=False)
    
    # Step 3: Build model
    model = build_lightweight_cnn(num_classes)
    model = compile_model(model)
    
    # Step 4: Initial training
    history = train_model(model, train_dataset, val_dataset)
    
    # Save initial model
    model.save(CONFIG["model_save_path"])
    print(f"\nInitial model saved to: {CONFIG['model_save_path']}")
    
    # Evaluate initial model
    print("\nEvaluating initial model on test set...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    print(f"Test Accuracy (initial): {test_acc * 100:.2f}%")
    
    # Step 5: Apply pruning
    pruned_model = apply_pruning(model, train_dataset, val_dataset)
    
    # Save pruned model
    pruned_model.save(CONFIG["pruned_model_path"])
    print(f"\nPruned model saved to: {CONFIG['pruned_model_path']}")
    
    # Evaluate pruned model
    print("\nEvaluating pruned model on test set...")
    pruned_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    pruned_loss, pruned_acc = pruned_model.evaluate(test_dataset, verbose=1)
    print(f"Test Accuracy (pruned): {pruned_acc * 100:.2f}%")
    
    # Step 6: Convert to TFLite with quantization
    convert_to_tflite(pruned_model, val_dataset)
    
    # Step 7: Evaluate TFLite models
    print("\n" + "=" * 60)
    print("Final Evaluation of TFLite Models")
    print("=" * 60)
    
    # Evaluate dynamic range quantized model
    dynamic_path = CONFIG["tflite_model_path"].replace(".tflite", "_dynamic.tflite")
    if os.path.exists(dynamic_path):
        evaluate_tflite_model(dynamic_path, test_dataset)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_df)}")
    print(f"Initial test accuracy: {test_acc * 100:.2f}%")
    print(f"Pruned test accuracy: {pruned_acc * 100:.2f}%")
    print("\nSaved models:")
    print(f"  - Keras model: {CONFIG['model_save_path']}")
    print(f"  - Pruned model: {CONFIG['pruned_model_path']}")
    print(f"  - TFLite models: {os.path.dirname(CONFIG['tflite_model_path'])}")
    print("\nModel is ready for mobile deployment!")


if __name__ == "__main__":
    main()
