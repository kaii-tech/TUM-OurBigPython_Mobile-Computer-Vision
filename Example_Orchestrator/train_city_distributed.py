#!/usr/bin/env python3
"""
Distributed City Locator training with MultiWorkerMirroredStrategy.
Designed to be launched via SSH from the orchestrator.
"""

import os
import json
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

try:
    import kagglehub  # Optional: only used when --data-dir is not provided
except ImportError:  # pragma: no cover
    kagglehub = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 380
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20  # stage 1 epochs
DEFAULT_FINE_TUNE_EPOCHS = 10
BASE_LR_STAGE1 = 1e-3
BASE_LR_STAGE2 = 1e-5
VALIDATION_SPLIT = 0.2
SEED = 42


def find_class_dir(root: Path, max_depth: int = 3, min_classes: int = 3):
    """Heuristic to find a directory that contains many class subfolders."""
    queue = [(root, 0)]
    best = None
    while queue:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if len(subdirs) >= min_classes:
            has_files = any(any(f.is_file() for f in d.iterdir()) for d in subdirs)
            if has_files:
                return current, subdirs
            best = best or (current, subdirs)
        for sd in subdirs:
            queue.append((sd, depth + 1))
    return best if best else (root, [d for d in root.iterdir() if d.is_dir()])


def load_datasets(data_root: Path, img_size: int, batch_size: int):
    """Create tf.data datasets with auto-sharding for distributed training."""
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    data_dir, class_dirs = find_class_dir(data_root)
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {data_root}")

    print(f"Using data directory: {data_dir}")
    print(f"Found {len(class_dirs)} class folders. Example: {[d.name for d in class_dirs[:10]]}")

    def make_ds(subset):
        ds = keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            seed=SEED,
            validation_split=VALIDATION_SPLIT,
            subset=subset,
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        return ds.with_options(options).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds("training")
    val_ds = make_ds("validation")

    num_classes = len(train_ds.class_names)
    print(f"Detected NUM_CLASSES = {num_classes}")
    return train_ds, val_ds, num_classes


def build_model(num_classes: int, img_size: int):
    """EfficientNetB4 backbone with the same head as City_Locator-BIG.ipynb."""
    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    base.trainable = False

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="efficientnetB4_city_locator")
    return model, base


def save_history(history, output_dir: Path, prefix: str):
    history_path = output_dir / f"{prefix}_history.json"
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with history_path.open("w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"History saved to {history_path}")


def convert_to_tflite(model, output_dir: Path, prefix: str):
    tflite_path = output_dir / f"{prefix}_float16.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with tflite_path.open("wb") as f:
        f.write(tflite_model)
    print(f"TFLite float16 saved to {tflite_path} ({tflite_path.stat().st_size / (1024 * 1024):.2f} MB)")


def is_chief(resolver):
    if resolver is None:
        return True
    task_type = resolver.task_type or "worker"
    task_id = resolver.task_id or 0
    return task_type == "worker" and task_id == 0


def main():
    parser = argparse.ArgumentParser(description="Distributed City Locator training")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory containing class subfolders; if omitted and kagglehub is available, downloads gsv-cities")
    parser.add_argument("--output-dir", type=str, default="./results", help="Where to save checkpoints and history")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Stage 1 (frozen backbone) epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-worker batch size")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    args = parser.parse_args()

    # Resolve data directory (download if needed)
    data_dir = args.data_dir
    if data_dir is None:
        if kagglehub is None:
            raise RuntimeError("kagglehub not installed and --data-dir not provided")
        print("Downloading gsv-cities via kagglehub...")
        data_dir = kagglehub.dataset_download("amaralibey/gsv-cities")
    data_root = Path(data_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy setup
    # Match notebook: enable mixed precision for speed/memory
    mixed_precision.set_global_policy("mixed_float16")

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    lr_stage1 = BASE_LR_STAGE1  # keep unscaled to mirror notebook
    lr_stage2 = BASE_LR_STAGE2

    print("=" * 70)
    print("City Locator Distributed Training")
    print("=" * 70)
    print(f"Workers in sync: {num_workers}")
    print(f"Stage1 LR: {lr_stage1}")
    print(f"Stage2 LR: {lr_stage2}")
    print(f"Batch size per worker: {args.batch_size}")
    print(f"Stage1 epochs: {args.epochs}")
    print(f"Stage2 epochs: {args.fine_tune_epochs}")
    print(f"Model: EfficientNetB4")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Data
    train_ds, val_ds, num_classes = load_datasets(data_root, args.img_size, args.batch_size)

    resolver = strategy.cluster_resolver
    chief = is_chief(resolver)

    with strategy.scope():
        model, base = build_model(num_classes, args.img_size)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        optimizer_stage1 = keras.optimizers.Adam(learning_rate=lr_stage1)
        model.compile(
            optimizer=optimizer_stage1,
            loss=loss,
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc")],
        )

    # Callbacks (checkpoint only on chief to avoid contention)
    # Stage 1: frozen backbone
    callbacks_stage1 = [keras.callbacks.BackupAndRestore(backup_dir=str(output_dir / "backup_stage1"))]
    if chief:
        ckpt_path = output_dir / "efficientnetb4_stage1_best.keras"
        callbacks_stage1.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    # Stage 2: unfreeze top ~25% (keep BatchNorm frozen)
    fine_tune_at = len(base.layers) * 3 // 4
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    with strategy.scope():
        optimizer_stage2 = keras.optimizers.Adam(learning_rate=lr_stage2)
        model.compile(
            optimizer=optimizer_stage2,
            loss=loss,
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc")],
        )

    callbacks_stage2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.BackupAndRestore(backup_dir=str(output_dir / "backup_stage2")),
    ]
    if chief:
        ckpt_path_ft = output_dir / "efficientnetb4_stage2_best.keras"
        callbacks_stage2.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path_ft),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    if chief:
        final_path = output_dir / "efficientnetb4_final.keras"
        model.save(final_path)
        print(f"Final model saved to {final_path}")
        save_history(history_stage1, output_dir, prefix="efficientnetb4_stage1")
        save_history(history_stage2, output_dir, prefix="efficientnetb4_stage2")
        convert_to_tflite(model, output_dir, prefix="efficientnetb4")

    print("Training complete")


if __name__ == "__main__":
    main()
