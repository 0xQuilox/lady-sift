import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original or converted model.")
    parser.add_argument("--data-dir", type=Path, default=Path("training/data/processed/test"))
    parser.add_argument("--model", type=Path, default=Path("training/output/best_model.h5"))
    parser.add_argument("--tfjs-model-dir", type=Path, default=Path("training/output/tfjs_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/output"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-tfjs", action="store_true", help="Round-trip TF.js model before evaluation.")
    return parser.parse_args()


def make_dataset(directory, image_size, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"],
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset.prefetch(AUTOTUNE)


def load_model(args):
    if not args.use_tfjs:
        return tf.keras.models.load_model(args.model)

    if not args.tfjs_model_dir.exists():
        raise FileNotFoundError(f"TF.js model directory not found: {args.tfjs_model_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix="lady_sift_tfjs_eval_"))
    try:
        keras_path = temporary_dir / "roundtrip.h5"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorflowjs.converters.converter",
                "--input_format=tfjs_layers_model",
                "--output_format=keras",
                str(args.tfjs_model_dir / "model.json"),
                str(keras_path),
            ],
            check=True,
        )
        return tf.keras.models.load_model(keras_path)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def collect_labels(dataset):
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy().reshape(-1).astype(int).tolist())
    return np.array(labels)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args.data_dir, args.image_size, args.batch_size)
    y_true = collect_labels(dataset)
    model = load_model(args)
    probabilities = model.predict(dataset).reshape(-1)
    y_pred = (probabilities >= args.threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "class_order": ["real", "fake"],
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) > 1 else None,
        "threshold": args.threshold,
        "evaluated_model": "tfjs_roundtrip" if args.use_tfjs else str(args.model),
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Confusion matrix rows=true, cols=pred, order=[real, fake]:")
    print(matrix)
    print(json.dumps(metrics, indent=2))
    if metrics["accuracy"] < 0.85:
        print("WARNING: Accuracy is below the 85% CIFAKE sanity target.")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
