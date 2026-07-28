import argparse
import json
from pathlib import Path

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 real/fake classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("training/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/output"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initial-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--initial-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_gpu_status():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("=" * 60)
        print("WARNING: No GPU detected. Training will run on CPU.")
        print("This will be significantly slower, potentially hours instead of minutes.")
        print("On native Windows, TensorFlow GPU support usually requires WSL2.")
        print("=" * 60)
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"Could not set memory growth on {gpu}: {error}")

    print("=" * 60)
    print(f"GPU(s) detected: {[gpu.name for gpu in gpus]}")
    print("Memory growth enabled to avoid TensorFlow over-allocating VRAM.")
    print("=" * 60)


def make_dataset(directory, image_size, batch_size, shuffle):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"],
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataset.prefetch(AUTOTUNE)


def build_model(image_size):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    augmented = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )(inputs)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    features = base_model(preprocessed, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    hidden = tf.keras.layers.Dense(128, activation="relu")(pooled)
    dropped = tf.keras.layers.Dropout(0.35)(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dropped)

    model = tf.keras.Model(inputs, outputs, name="lady_sift_mobilenetv2")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def callbacks(output_dir, checkpoint_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / checkpoint_name),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"), append=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
    ]


def unfreeze_top_layers(base_model, fine_tune_layers):
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - fine_tune_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def merge_histories(first, second):
    merged = {}
    for history in (first.history, second.history):
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def best_validation_accuracy(history):
    values = history.history.get("val_accuracy", [])
    return max(values) if values else float("-inf")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    log_gpu_status()

    train_ds = make_dataset(args.data_dir / "train", args.image_size, args.batch_size, True)
    val_ds = make_dataset(args.data_dir / "val", args.image_size, args.batch_size, False)

    model, base_model = build_model(args.image_size)
    compile_model(model, args.initial_lr)

    history_one = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        callbacks=callbacks(args.output_dir, "best_frozen_model.h5"),
    )

    unfreeze_top_layers(base_model, args.fine_tune_layers)
    compile_model(model, args.fine_tune_lr)
    total_epochs = args.initial_epochs + args.fine_tune_epochs
    history_two = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_one.epoch),
        epochs=total_epochs,
        callbacks=callbacks(args.output_dir, "best_finetuned_model.h5"),
    )

    best_path = args.output_dir / "best_model.h5"
    frozen_best = args.output_dir / "best_frozen_model.h5"
    finetuned_best = args.output_dir / "best_finetuned_model.h5"
    source_best = finetuned_best
    if best_validation_accuracy(history_one) > best_validation_accuracy(history_two):
        source_best = frozen_best
    tf.io.gfile.copy(str(source_best), str(best_path), overwrite=True)

    final_path = args.output_dir / "final_model.h5"
    model.save(final_path)

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(merge_histories(history_one, history_two), handle, indent=2)

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
