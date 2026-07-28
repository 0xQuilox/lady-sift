import argparse
import csv
import os
import random
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_NAMES = {"real", "0"}
FAKE_NAMES = {"fake", "ai", "generated", "synthetic", "1"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare real/fake image data splits.")
    parser.add_argument("--source", type=Path, default=None, help="Existing dataset directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="CSV with filepath,label columns.")
    parser.add_argument(
        "--kaggle-dataset",
        default="birdy654/cifake-real-and-ai-generated-synthetic-images",
        help="Kaggle dataset slug to download when --source is omitted.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("training/data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hard-linking/symlinking. Safer but uses more disk.",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "undersample"],
        default="none",
        help="Optional class balancing before split.",
    )
    return parser.parse_args()


def normalize_label(label):
    normalized = str(label).strip().lower()
    if normalized in REAL_NAMES:
        return "real"
    if normalized in FAKE_NAMES:
        return "fake"
    raise ValueError(f"Unsupported label: {label!r}. Use real/fake/ai or 0/1.")


def download_kaggle_dataset(slug, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.rglob("*")):
        print(f"Using existing raw dataset at {raw_dir}")
        return raw_dir

    print(f"Downloading Kaggle dataset {slug} to {raw_dir}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"],
        check=True,
    )

    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw_dir)
        archive.unlink()
    return raw_dir


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def collect_from_manifest(manifest_path):
    samples = []
    base_dir = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filepath", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain filepath,label columns.")
        for row in reader:
            path = Path(row["filepath"])
            if not path.is_absolute():
                path = base_dir / path
            samples.append((path, normalize_label(row["label"])))
    return samples


def collect_from_directories(source_dir):
    samples = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = None
        for part in reversed(path.parts):
            lower = part.lower()
            if lower in REAL_NAMES:
                label = "real"
                break
            if lower in FAKE_NAMES:
                label = "fake"
                break
        if label:
            samples.append((path, label))
    return samples


def validate_samples(samples):
    valid = []
    skipped = []
    for path, label in tqdm(samples, desc="Validating images"):
        if path.exists() and is_valid_image(path):
            valid.append((path, label))
        else:
            skipped.append(str(path))
    if skipped:
        print(f"Skipped {len(skipped)} missing/corrupt images.")
    return valid


def balance_samples(samples, strategy, rng):
    if strategy == "none":
        return samples

    by_label = {"real": [], "fake": []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    target = min(len(items) for items in by_label.values())
    balanced = []
    for items in by_label.values():
        rng.shuffle(items)
        balanced.extend(items[:target])
    rng.shuffle(balanced)
    return balanced


def stratified_split(samples, val_ratio, test_ratio, rng):
    splits = {"train": [], "val": [], "test": []}
    by_label = {"real": [], "fake": []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    for label, label_samples in by_label.items():
        rng.shuffle(label_samples)
        total = len(label_samples)
        test_count = int(total * test_ratio)
        val_count = int(total * val_ratio)
        splits["test"].extend(label_samples[:test_count])
        splits["val"].extend(label_samples[test_count : test_count + val_count])
        splits["train"].extend(label_samples[test_count + val_count :])

    for split_samples in splits.values():
        rng.shuffle(split_samples)
    return splits


def link_or_copy(source, destination, copy):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def write_split(splits, output_dir, copy):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split_name, split_samples in splits.items():
        seen_names = Counter()
        for source, label in tqdm(split_samples, desc=f"Writing {split_name}"):
            seen_names[source.name] += 1
            suffix = "" if seen_names[source.name] == 1 else f"_{seen_names[source.name]}"
            destination_name = f"{source.stem}{suffix}{source.suffix.lower()}"
            destination = output_dir / split_name / label / destination_name
            link_or_copy(source, destination, copy)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.manifest:
        samples = collect_from_manifest(args.manifest)
    else:
        source = args.source or download_kaggle_dataset(args.kaggle_dataset, args.raw_dir)
        samples = collect_from_directories(source)

    if not samples:
        raise RuntimeError("No labeled images found. Expected real/ and fake/ or ai/ directories.")

    samples = validate_samples(samples)
    samples = balance_samples(samples, args.balance, rng)
    counts = Counter(label for _, label in samples)
    print(f"Usable samples: {dict(counts)}")

    if set(counts) != {"real", "fake"}:
        raise RuntimeError("Both real and fake classes are required.")

    splits = stratified_split(samples, args.val_ratio, args.test_ratio, rng)
    write_split(splits, args.output_dir, args.copy)

    for split_name, split_samples in splits.items():
        split_counts = Counter(label for _, label in split_samples)
        print(f"{split_name}: {dict(split_counts)}")
    print(f"Prepared data at {args.output_dir}")


if __name__ == "__main__":
    main()
