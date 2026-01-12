"""
Lightweight CNN for Room/Entrance Classification (Mobile Target)
================================================================
A complete pipeline for training, pruning, and quantizing a CNN for mobile deployment.
Classifies indoor rooms and entrances from images.

Tech Stack: Python, TensorFlow/Keras, NumPy
Target: TFLite model optimized for mobile devices

Dataset Structure:
    Lab_Course_Dataset/
        Building_Address/
            Building_Address/
                Indoor/
                    Floor/
                        RoomID/
                            images...
                Outdoor/
                    images...
"""

import os
# Set environment variable to use tf_keras with TFMOT before importing TensorFlow
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning messages

import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from collections import defaultdict
import random

tf.get_logger().setLevel(logging.ERROR)  # Only show errors, not warnings

# Use tf_keras for compatibility with tensorflow_model_optimization
import tf_keras as keras
from tf_keras import layers, models, callbacks, optimizers, regularizers
import tensorflow_model_optimization as tfmot

# =============================================================================
# GPU Configuration (call early, before any TF operations)
# =============================================================================
def configure_gpu():
    """Configure GPU settings for optimal performance with CUDA on Windows/WSL."""
    print("\n" + "=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    
    # Check if TensorFlow was built with CUDA
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List all physical devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"CPUs available: {len(cpus)}")
    print(f"GPUs available: {len(gpus)}")
    
    if gpus:
        print(f"\nFound {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")
        
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n[OK] Memory growth enabled")
        except RuntimeError as e:
            print(f"[WARNING] Memory growth setting failed: {e}")
        
        # Try to enable mixed precision for faster training
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[OK] Mixed precision (float16) enabled")
        except Exception as e:
            print(f"[INFO] Mixed precision not enabled: {e}")
        
        # Verify GPU is actually usable with a simple operation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print("[OK] GPU computation test passed")
        except Exception as e:
            print(f"[ERROR] GPU computation failed: {e}")
            return False
        
        print("\n" + "-" * 40)
        print("GPU is ready for training!")
        print("-" * 40)
        return True
    else:
        print("\n[WARNING] No GPU detected!")
        print("Continuing with CPU...")
        return False

# Configure GPU at module load time
GPU_AVAILABLE = configure_gpu()

# =============================================================================
# Configuration
# =============================================================================
# Detect if running in WSL or Windows
import platform
if 'microsoft' in platform.uname().release.lower() or 'wsl' in platform.uname().release.lower():
    # WSL paths
    BASE_PATH = "/mnt/c/Users/Klein/Desktop/Labcourse"
else:
    # Windows paths
    BASE_PATH = r"C:\Users\Klein\Desktop\Labcourse"

CONFIG = {
    # Data paths
    "data_dir": os.path.join(BASE_PATH, "Lab_Course_Dataset"),
    
    # Image settings - 224x224 for mobile compatibility
    "img_height": 224,
    "img_width": 224,
    "channels": 3,
    
    # Data split ratios
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Training settings - optimized for ~6100 images, 13 classes
    # Batch 64 = ~76 steps/epoch with 4880 training images
    "batch_size": 64 if GPU_AVAILABLE else 32,
    "epochs_initial": 100,  # More epochs, early stopping will halt when converged
    "epochs_pruning": 15,
    "learning_rate": 0.001,  # Higher LR for larger batch (linear scaling)
    
    # Regularization - balanced for medium dataset
    "dropout_conv": 0.25,  # Lighter conv dropout
    "dropout_dense": 0.4,  # Moderate dense dropout
    "l2_reg": 0.0005,  # Lighter L2 for better convergence
    
    # Pruning settings
    "pruning_initial_sparsity": 0.0,
    "pruning_final_sparsity": 0.5,
    
    # Output paths
    "model_save_path": os.path.join(BASE_PATH, "Labcourse AI", "room_model.keras"),
    "pruned_model_path": os.path.join(BASE_PATH, "Labcourse AI", "room_pruned.keras"),
    "tflite_model_path": os.path.join(BASE_PATH, "Labcourse AI", "room_model.tflite"),
    "tflite_quantized_path": os.path.join(BASE_PATH, "Labcourse AI", "room_quantized.tflite"),
}

# =============================================================================
# Data Pipeline
# =============================================================================
def discover_dataset():
    """
    Discover all images and their labels from the folder structure.
    
    Handles multiple folder structures:
    - Building/Indoor/Floor/RoomID/images
    - Building/Building/Indoor/Floor/RoomID/images  
    - Building/Indoor/RoomID/images (no floor level)
    - Building/Outdoor/Direction/images
    - Building/Building/Outdoor/Direction/images
    
    Label = RoomID (e.g., HW_716, RM_126) or Outdoor_Direction
    """
    print("=" * 60)
    print("Discovering dataset from folder structure...")
    print("=" * 60)
    
    data_dir = Path(CONFIG["data_dir"])
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Collect all images with their labels
    images_by_class = defaultdict(list)
    
    def collect_images_from_dir(directory, label):
        """Recursively collect all images from a directory."""
        count = 0
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                images_by_class[label].append(str(item))
                count += 1
            elif item.is_dir():
                # Recurse into subdirectories
                count += collect_images_from_dir(item, label)
        return count
    
    def find_indoor_outdoor_dirs(start_dir):
        """Find Indoor and Outdoor directories, handling nested structures."""
        indoor_dirs = []
        outdoor_dirs = []
        
        # Check direct children
        for item in start_dir.iterdir():
            if item.is_dir():
                if item.name == 'Indoor':
                    indoor_dirs.append(item)
                elif item.name == 'Outdoor':
                    outdoor_dirs.append(item)
                else:
                    # Check one level deeper (nested building structure)
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            if subitem.name == 'Indoor':
                                indoor_dirs.append(subitem)
                            elif subitem.name == 'Outdoor':
                                outdoor_dirs.append(subitem)
        
        return indoor_dirs, outdoor_dirs
    
    def process_indoor_dir(indoor_dir):
        """Process Indoor directory - find all room folders (HW_*, RM_*)."""
        for item in indoor_dir.iterdir():
            if not item.is_dir():
                continue
            
            # Check if this is a room folder directly (HW_*, RM_*)
            if item.name.startswith(('HW_', 'RM_')):
                label = item.name
                collect_images_from_dir(item, label)
            else:
                # This might be a floor folder - check its contents
                for subitem in item.iterdir():
                    if subitem.is_dir() and subitem.name.startswith(('HW_', 'RM_')):
                        label = subitem.name
                        collect_images_from_dir(subitem, label)
    
    def process_outdoor_dir(outdoor_dir, building_name):
        """Process Outdoor directory - each subdirectory is a view direction."""
        # Check for images directly in Outdoor folder
        for item in outdoor_dir.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                label = f"Outdoor_{building_name}"
                images_by_class[label].append(str(item))
            elif item.is_dir():
                # Subdirectory like North, South, Main_entrance
                label = f"Outdoor_{item.name}"
                collect_images_from_dir(item, label)
    
    # Iterate through buildings
    for building_dir in data_dir.iterdir():
        if not building_dir.is_dir():
            continue
        
        building_name = building_dir.name.split('_')[0]  # Get street name part
        
        # Find Indoor and Outdoor directories (handles nested structures)
        indoor_dirs, outdoor_dirs = find_indoor_outdoor_dirs(building_dir)
        
        # Process all Indoor directories found
        for indoor_dir in indoor_dirs:
            process_indoor_dir(indoor_dir)
        
        # Process all Outdoor directories found
        for outdoor_dir in outdoor_dirs:
            process_outdoor_dir(outdoor_dir, building_name)
    
    # Print statistics
    print(f"\nFound {len(images_by_class)} classes:")
    total_images = 0
    for label, images in sorted(images_by_class.items()):
        print(f"  {label}: {len(images)} images")
        total_images += len(images)
    print(f"\nTotal images: {total_images}")
    
    return images_by_class


def create_train_val_test_split(images_by_class, seed=42):
    """
    Split the dataset into train, validation, and test sets.
    Stratified split to maintain class distribution.
    """
    print("\n" + "=" * 60)
    print("Creating train/validation/test splits...")
    print("=" * 60)
    
    random.seed(seed)
    np.random.seed(seed)
    
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []
    
    # Create label encoding
    all_labels = sorted(images_by_class.keys())
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Split each class
    for label, images in images_by_class.items():
        random.shuffle(images)
        n = len(images)
        
        # Calculate split indices
        train_end = int(n * CONFIG["train_ratio"])
        val_end = train_end + int(n * CONFIG["val_ratio"])
        
        # Ensure at least 1 sample in each split if possible
        if n >= 3:
            train_end = max(1, train_end)
            val_end = max(train_end + 1, val_end)
        
        # Split
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Add to lists
        label_idx = label_to_idx[label]
        train_images.extend(train_imgs)
        train_labels.extend([label_idx] * len(train_imgs))
        val_images.extend(val_imgs)
        val_labels.extend([label_idx] * len(val_imgs))
        test_images.extend(test_imgs)
        test_labels.extend([label_idx] * len(test_imgs))
    
    print(f"Train samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Number of classes: {len(label_to_idx)}")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), label_to_idx, idx_to_label


def create_tf_dataset(image_paths, labels, is_training=False):
    """Create a tf.data.Dataset with efficient GPU prefetching."""
    
    def load_and_preprocess_image(path, label):
        """Load image from path and preprocess."""
        # Read file
        img = tf.io.read_file(path)
        # Decode image (handles jpg, jpeg, png)
        img = tf.image.decode_image(img, channels=CONFIG["channels"], expand_animations=False)
        img.set_shape([None, None, CONFIG["channels"]])
        # Resize
        img = tf.image.resize(img, [CONFIG["img_height"], CONFIG["img_width"]])
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    @tf.function
    def augment(image, label):
        """Apply balanced data augmentation for transfer learning (GPU-accelerated)."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation simulation via crop and resize
        crop_fraction = tf.random.uniform([], minval=0.80, maxval=1.0)
        crop_h = tf.cast(tf.cast(CONFIG["img_height"], tf.float32) * crop_fraction, tf.int32)
        crop_w = tf.cast(tf.cast(CONFIG["img_width"], tf.float32) * crop_fraction, tf.int32)
        image = tf.image.random_crop(image, [crop_h, crop_w, CONFIG["channels"]])
        image = tf.image.resize(image, [CONFIG["img_height"], CONFIG["img_width"]])
        
        # Color augmentations - balanced for indoor room images
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 10000), reshuffle_each_iteration=True)
    
    # Load and preprocess images (parallel)
    num_workers = tf.data.AUTOTUNE
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=num_workers,
        deterministic=False
    )
    
    # Cache validation/test sets
    if not is_training:
        dataset = dataset.cache()
    
    # Apply augmentation if training
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=num_workers, deterministic=False)
    
    # Batch and prefetch
    dataset = dataset.batch(CONFIG["batch_size"], drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # GPU pipeline optimization
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = 8
    dataset = dataset.with_options(options)
    
    return dataset


# =============================================================================
# Model Architecture
# =============================================================================
def build_mobilenet_transfer(num_classes):
    """
    Build a MobileNetV2 transfer learning model for room classification.
    Uses ImageNet pretrained weights for excellent generalization.
    Optimized for mobile deployment.
    """
    print("\n" + "=" * 60)
    print("Building MobileNetV2 Transfer Learning Model...")
    print("=" * 60)
    
    # Load MobileNetV2 with ImageNet weights (no top classifier)
    base_model = keras.applications.MobileNetV2(
        input_shape=(CONFIG["img_height"], CONFIG["img_width"], CONFIG["channels"]),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Full MobileNetV2 (can use 0.5 or 0.75 for smaller)
    )
    
    # Freeze base model initially (train only the head)
    base_model.trainable = False
    
    l2_reg = regularizers.l2(CONFIG["l2_reg"])
    
    # Build the full model
    inputs = keras.Input(shape=(CONFIG["img_height"], CONFIG["img_width"], CONFIG["channels"]))
    
    # MobileNetV2 expects inputs in [-1, 1] range (preprocess_input)
    # Our data is [0, 1], so rescale
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head with dropout - larger head for more capacity
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg, name='dense1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout_dense"])(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg, name='dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout_dense"])(x)
    
    # Output layer - use float32 for numerical stability with mixed precision
    outputs = layers.Dense(num_classes, dtype='float32', activation='softmax', name='output')(x)
    
    model = keras.Model(inputs, outputs, name='room_classifier_mobilenet')
    model.summary()
    
    # Store base model reference for fine-tuning later
    model.base_model = base_model
    
    print(f"\nMobileNetV2 base: {len(base_model.layers)} layers (frozen)")
    print(f"Model will run on: {'/GPU:0' if GPU_AVAILABLE else '/CPU:0'}")
    
    return model


def unfreeze_model(model, unfreeze_layers=50):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    Called after initial training of the classification head.
    """
    print(f"\n[Fine-tuning] Unfreezing top {unfreeze_layers} layers of MobileNetV2...")
    
    base_model = model.base_model
    base_model.trainable = True
    
    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    trainable_count = sum([layer.trainable for layer in base_model.layers])
    print(f"[Fine-tuning] {trainable_count} base layers now trainable")
    
    return model


def build_lightweight_cnn(num_classes):
    """
    Build a lightweight CNN optimized for mobile deployment.
    NOTE: For small datasets, use build_mobilenet_transfer() instead.
    """
    print("\n" + "=" * 60)
    print("Building Lightweight CNN Architecture...")
    print("=" * 60)
    
    l2_reg = regularizers.l2(CONFIG["l2_reg"])
    
    model = models.Sequential([
        # Input layer
        layers.InputLayer(input_shape=(CONFIG["img_height"], CONFIG["img_width"], CONFIG["channels"])),
        
        # Block 1: Initial feature extraction
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=l2_reg, name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(CONFIG["dropout_conv"]),
        
        # Block 2: Deeper features
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2_reg, name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(CONFIG["dropout_conv"]),
        
        # Block 3: More complex patterns
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2_reg, name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(CONFIG["dropout_conv"]),
        
        # Block 4: High-level features
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2_reg, name='conv4'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(CONFIG["dropout_conv"]),
        
        # Block 5: Final conv block with depthwise separable for efficiency
        layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu',
                               depthwise_regularizer=l2_reg,
                               pointwise_regularizer=l2_reg, name='sep_conv5'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Classification head
        layers.Dense(256, activation='relu', kernel_regularizer=l2_reg, name='dense1'),
        layers.BatchNormalization(),
        layers.Dropout(CONFIG["dropout_dense"]),
        
        layers.Dense(128, activation='relu', kernel_regularizer=l2_reg, name='dense2'),
        layers.Dropout(CONFIG["dropout_dense"]),
        
        # Output layer - use float32 for numerical stability with mixed precision
        layers.Dense(num_classes, dtype='float32', activation='softmax', name='output')
    ])
    
    model.summary()
    
    print(f"\nModel will run on: {'/GPU:0' if GPU_AVAILABLE else '/CPU:0'}")
    
    return model


def compile_model(model, learning_rate=None, use_label_smoothing=True):
    """Compile the model with optimizer and loss function optimized for GPU."""
    lr = learning_rate if learning_rate else CONFIG["learning_rate"]
    optimizer = optimizers.Adam(learning_rate=lr)
    
    # With batch 64, use 64 steps per execution for maximum GPU throughput
    steps_per_exec = 64 if GPU_AVAILABLE else 1
    
    # Use label smoothing to improve generalization (reduces overconfidence)
    if use_label_smoothing:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # Note: Label smoothing is applied differently for sparse labels
        # We'll use a custom approach in training or rely on strong regularization
    else:
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=GPU_AVAILABLE,
        steps_per_execution=steps_per_exec
    )
    
    if GPU_AVAILABLE:
        print(f"\n[GPU Optimizations Enabled]")
        print(f"  - XLA JIT compilation: ON")
        print(f"  - Steps per execution: {steps_per_exec}")
        print(f"  - Mixed precision: float16")
        print(f"  - Learning rate: {lr}")
    
    return model


# =============================================================================
# Training
# =============================================================================
def train_model(model, train_dataset, val_dataset, is_transfer_learning=True):
    """Train the model with callbacks and GPU-optimized settings."""
    print("\n" + "=" * 60)
    print("Training Model...")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        print(f"\nTraining with batch_size={CONFIG['batch_size']} on GPU")
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=os.path.join(BASE_PATH, 'logs', 'room_classification'),
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=0
    )
    
    # Model checkpoint to save best model
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=CONFIG["model_save_path"],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    if is_transfer_learning and hasattr(model, 'base_model'):
        # Two-stage training for transfer learning
        print("\n--- Stage 1: Training classification head (base frozen) ---")
        
        # Stage 1: Train only the classification head (more epochs for better head)
        history1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=30,  # More epochs for better head initialization
            callbacks=[early_stopping, reduce_lr, tensorboard_cb, checkpoint_cb],
            verbose=1
        )
        
        # Stage 2: Fine-tune more layers of base model
        print("\n--- Stage 2: Fine-tuning with unfrozen base layers ---")
        
        # Unfreeze more layers for better fine-tuning (100 instead of 50)
        model = unfreeze_model(model, unfreeze_layers=100)
        
        # Recompile with lower learning rate for fine-tuning
        optimizer = optimizers.Adam(learning_rate=CONFIG["learning_rate"] * 0.05)  # Even lower LR
        steps_per_exec = 64 if GPU_AVAILABLE else 1
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=GPU_AVAILABLE,
            steps_per_execution=steps_per_exec
        )
        
        # Reset callbacks for stage 2
        early_stopping_ft = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr_ft = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        history2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG["epochs_initial"],
            callbacks=[early_stopping_ft, reduce_lr_ft, tensorboard_cb, checkpoint_cb],
            verbose=1
        )
        
        # Combine histories
        history = history2  # Return fine-tuning history
    else:
        # Standard training (no transfer learning)
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG["epochs_initial"],
            callbacks=[early_stopping, reduce_lr, tensorboard_cb, checkpoint_cb],
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
    
    # Disable mixed precision for pruning (TFMOT doesn't support float16)
    original_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    print("[INFO] Temporarily disabled mixed precision for pruning compatibility")
    
    # For transfer learning models, we need to save and reload to get a clean model
    # This removes the base_model attribute and ensures float32 weights
    temp_model_path = CONFIG["model_save_path"].replace(".keras", "_temp_for_pruning.keras")
    model.save(temp_model_path)
    model_for_pruning = keras.models.load_model(temp_model_path)
    
    # Clean up temp file
    import os as os_module
    if os_module.path.exists(temp_model_path):
        os_module.remove(temp_model_path)
    
    num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    end_step = num_batches * CONFIG["epochs_pruning"]
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=CONFIG["pruning_initial_sparsity"],
            final_sparsity=CONFIG["pruning_final_sparsity"],
            begin_step=0,
            end_step=end_step
        )
    }
    
    def apply_pruning_to_layer(layer):
        # Skip layers that shouldn't be pruned
        skip_layers = (
            layers.BatchNormalization,
            layers.Dropout,
            layers.GlobalAveragePooling2D,
            layers.MaxPooling2D,
            layers.Flatten,
            layers.InputLayer,
            layers.Rescaling,
        )
        if isinstance(layer, skip_layers):
            return layer
        # Skip the output layer to preserve class predictions
        if hasattr(layer, 'name') and layer.name == 'output':
            return layer
        # Only prune Dense and Conv layers
        if isinstance(layer, (layers.Dense, layers.Conv2D, layers.SeparableConv2D)):
            try:
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            except Exception as e:
                print(f"[WARNING] Could not prune layer {layer.name}: {e}")
                return layer
        return layer
    
    try:
        pruned_model = keras.models.clone_model(
            model_for_pruning,
            clone_function=apply_pruning_to_layer
        )
    except Exception as e:
        print(f"[WARNING] Model cloning for pruning failed: {e}")
        print("[INFO] Skipping pruning, returning original model")
        tf.keras.mixed_precision.set_global_policy(original_policy)
        return model
    
    # Copy weights from original model
    try:
        pruned_model.set_weights(model_for_pruning.get_weights())
    except Exception as e:
        print(f"[WARNING] Weight transfer failed: {e}")
    
    pruned_model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"] * 0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs'),
    ]
    
    print(f"Fine-tuning with pruning for {CONFIG['epochs_pruning']} epochs...")
    
    try:
        pruned_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG["epochs_pruning"],
            callbacks=pruning_callbacks,
            verbose=1
        )
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    except Exception as e:
        print(f"[WARNING] Pruning training failed: {e}")
        print("[INFO] Returning unpruned model")
        tf.keras.mixed_precision.set_global_policy(original_policy)
        return model
    
    # Restore mixed precision policy
    tf.keras.mixed_precision.set_global_policy(original_policy)
    print("[INFO] Restored mixed precision policy")
    
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
    
    # Disable mixed precision for TFLite conversion
    original_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    print("[INFO] Temporarily disabled mixed precision for TFLite conversion")
    
    # Save and reload model to ensure float32 weights
    temp_model_path = CONFIG["model_save_path"].replace(".keras", "_temp_for_tflite.keras")
    model.save(temp_model_path)
    model_for_conversion = keras.models.load_model(temp_model_path)
    
    # Clean up temp file
    import os as os_module
    if os_module.path.exists(temp_model_path):
        os_module.remove(temp_model_path)
    
    # Create representative dataset generator with proper float32 casting
    def representative_dataset_gen():
        count = 0
        for images, _ in val_dataset.take(100):  # Take more samples for better calibration
            for i in range(min(images.shape[0], 10)):  # Limit per batch
                if count >= 200:  # Total limit
                    return
                # Ensure float32 for quantization calibration
                img = tf.cast(tf.expand_dims(images[i], axis=0), tf.float32)
                yield [img]
                count += 1
    
    # 1. Dynamic Range Quantization
    print("\n1. Creating Dynamic Range Quantized model...")
    try:
        converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model_for_conversion)
        converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_dynamic = converter_dynamic.convert()
        
        dynamic_path = CONFIG["tflite_model_path"].replace(".tflite", "_dynamic.tflite")
        with open(dynamic_path, 'wb') as f:
            f.write(tflite_dynamic)
        print(f"   Saved to: {dynamic_path}")
        print(f"   Size: {len(tflite_dynamic) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   Dynamic quantization failed: {e}")
        tflite_dynamic = None
    
    # 2. Full Integer Quantization
    print("\n2. Creating Full Integer Quantized model...")
    try:
        converter_int = tf.lite.TFLiteConverter.from_keras_model(model_for_conversion)
        converter_int.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int.representative_dataset = representative_dataset_gen
        converter_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int.inference_input_type = tf.uint8
        converter_int.inference_output_type = tf.uint8
        
        tflite_int = converter_int.convert()
        int_path = CONFIG["tflite_quantized_path"]
        with open(int_path, 'wb') as f:
            f.write(tflite_int)
        print(f"   Saved to: {int_path}")
        print(f"   Size: {len(tflite_int) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   Full integer quantization failed: {e}")
        print("   Falling back to float16 quantization...")
        
        try:
            converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model_for_conversion)
            converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
            converter_fp16.target_spec.supported_types = [tf.float16]
            tflite_fp16 = converter_fp16.convert()
            
            fp16_path = CONFIG["tflite_quantized_path"].replace(".tflite", "_fp16.tflite")
            with open(fp16_path, 'wb') as f:
                f.write(tflite_fp16)
            print(f"   Saved to: {fp16_path}")
            print(f"   Size: {len(tflite_fp16) / 1024 / 1024:.2f} MB")
        except Exception as e2:
            print(f"   Float16 quantization also failed: {e2}")
    
    # 3. Standard TFLite (no quantization)
    print("\n3. Creating Standard TFLite model (no quantization)...")
    try:
        converter_std = tf.lite.TFLiteConverter.from_keras_model(model_for_conversion)
        tflite_std = converter_std.convert()
        
        std_path = CONFIG["tflite_model_path"]
        with open(std_path, 'wb') as f:
            f.write(tflite_std)
        print(f"   Saved to: {std_path}")
        print(f"   Size: {len(tflite_std) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   Standard conversion failed: {e}")
    
    # Restore mixed precision policy
    tf.keras.mixed_precision.set_global_policy(original_policy)
    print("\n[INFO] Restored mixed precision policy")
    
    return tflite_dynamic


def evaluate_tflite_model(tflite_model_path, test_dataset):
    """Evaluate TFLite model accuracy."""
    print(f"\nEvaluating TFLite model: {tflite_model_path}")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input dtype: {input_details[0]['dtype']}")
        print(f"   Output dtype: {output_details[0]['dtype']}")
        
        correct = 0
        total = 0
        
        for images, labels in test_dataset:
            for i in range(images.shape[0]):
                # Prepare input data
                input_data = tf.expand_dims(images[i], axis=0)
                
                # Handle quantized models (uint8 input)
                if input_details[0]['dtype'] == np.uint8:
                    input_scale = input_details[0]['quantization'][0]
                    input_zero_point = input_details[0]['quantization'][1]
                    if input_scale != 0:
                        input_data = input_data / input_scale + input_zero_point
                    input_data = tf.clip_by_value(input_data, 0, 255)
                    input_data = tf.cast(input_data, tf.uint8)
                else:
                    # Ensure float32 for non-quantized models
                    input_data = tf.cast(input_data, tf.float32)
                
                interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted = np.argmax(output_data)
                
                if predicted == labels[i].numpy():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"   TFLite Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    except Exception as e:
        print(f"   Evaluation failed: {e}")
        return 0.0


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """Execute the complete training and optimization pipeline."""
    print("\n" + "=" * 60)
    print("ROOM/ENTRANCE CLASSIFICATION - MOBILE OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        print(f"\nUsing GPU acceleration with mixed precision")
    else:
        print("\nRunning on CPU (no GPU detected)")
    
    # Step 1: Discover dataset from folder structure
    images_by_class = discover_dataset()
    
    if not images_by_class:
        print("ERROR: No images found in dataset directory!")
        print(f"Expected path: {CONFIG['data_dir']}")
        return
    
    # Step 2: Create train/val/test splits
    (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls), label_to_idx, idx_to_label = \
        create_train_val_test_split(images_by_class)
    
    num_classes = len(label_to_idx)
    
    # Save label mappings for inference
    label_mapping_path = os.path.join(
        os.path.dirname(CONFIG["model_save_path"]), 
        "room_label_mapping.npy"
    )
    np.save(label_mapping_path, idx_to_label)
    print(f"Label mapping saved to: {label_mapping_path}")
    
    # Step 3: Create tf.data datasets
    print("\nCreating tf.data pipelines...")
    train_dataset = create_tf_dataset(train_imgs, train_lbls, is_training=True)
    val_dataset = create_tf_dataset(val_imgs, val_lbls, is_training=False)
    test_dataset = create_tf_dataset(test_imgs, test_lbls, is_training=False)
    
    # Step 4: Build model - Use MobileNetV2 transfer learning for small datasets
    # For 2778 images and 13 classes, transfer learning is essential
    print("\n" + "-" * 40)
    print("Using MobileNetV2 Transfer Learning (recommended for small datasets)")
    print("-" * 40)
    model = build_mobilenet_transfer(num_classes)
    model = compile_model(model)
    
    # Step 5: Initial training (two-stage for transfer learning)
    history = train_model(model, train_dataset, val_dataset, is_transfer_learning=True)
    
    # Load best model from checkpoint
    model = keras.models.load_model(CONFIG["model_save_path"])
    
    # Evaluate initial model
    print("\nEvaluating initial model on test set...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    print(f"Test Accuracy (initial): {test_acc * 100:.2f}%")
    
    # Step 6: Apply pruning
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
    
    # Step 7: Convert to TFLite with quantization
    convert_to_tflite(pruned_model, val_dataset)
    
    # Step 8: Evaluate TFLite models
    print("\n" + "=" * 60)
    print("Final Evaluation of TFLite Models")
    print("=" * 60)
    
    dynamic_path = CONFIG["tflite_model_path"].replace(".tflite", "_dynamic.tflite")
    if os.path.exists(dynamic_path):
        evaluate_tflite_model(dynamic_path, test_dataset)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_imgs)}")
    print(f"Validation samples: {len(val_imgs)}")
    print(f"Test samples: {len(test_imgs)}")
    print(f"Initial test accuracy: {test_acc * 100:.2f}%")
    print(f"Pruned test accuracy: {pruned_acc * 100:.2f}%")
    print("\nSaved models:")
    print(f"  - Keras model: {CONFIG['model_save_path']}")
    print(f"  - Pruned model: {CONFIG['pruned_model_path']}")
    print(f"  - TFLite models: {os.path.dirname(CONFIG['tflite_model_path'])}")
    print("\nClass labels:")
    for idx, label in sorted(idx_to_label.items()):
        print(f"  {idx}: {label}")
    print("\nModel is ready for mobile deployment!")


if __name__ == "__main__":
    main()
