import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Keras model to TensorFlow.js.")
    parser.add_argument("--model", type=Path, default=Path("training/output/best_model.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/output/tfjs_model"))
    parser.add_argument(
        "--quantization-bytes",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 enables uint8 weight quantization; 2 enables uint16.",
    )
    return parser.parse_args()


def directory_size(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use current interpreter to avoid PATH mismatch (e.g. global 3.13 vs py311 venv, Colab vs Kaggle)
    command = [
        sys.executable,
        "-m",
        "tensorflowjs.converters.converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        f"--quantization_bytes={args.quantization_bytes}",
        str(args.model),
        str(args.output_dir),
    ]
    subprocess.run(command, check=True)

    size_bytes = directory_size(args.output_dir)
    print(f"TF.js model written to {args.output_dir}")
    print(f"TF.js model size: {size_bytes / (1024 * 1024):.2f} MB")
    if size_bytes > 10 * 1024 * 1024:
        print("WARNING: TF.js model exceeds the 10MB target.")


if __name__ == "__main__":
    main()
