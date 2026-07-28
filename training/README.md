# Lady Sift Training Pipeline

This folder trains a lightweight MobileNetV2 binary classifier for real vs AI-generated images and exports it to TensorFlow.js for the future Chrome extension.

## Environment

Use Python 3.10 or 3.11 on a Colab/Kaggle free-tier GPU.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r training/requirements.txt
```

For Kaggle downloads, configure `~/.kaggle/kaggle.json` or upload it in Colab before running `prepare_data.py`.

## 1. Prepare Data

Default CIFAKE download:

```bash
python training/prepare_data.py
```

Existing directory with `real/` and `fake/` or `ai/` folders anywhere below it:

```bash
python training/prepare_data.py --source /path/to/dataset
```

CSV manifest with `filepath,label` columns:

```bash
python training/prepare_data.py --manifest /path/to/manifest.csv
```

Outputs stratified `80/10/10` splits to `training/data/processed/{train,val,test}/{real,fake}`. Corrupt or unreadable images are skipped. Use `--balance undersample` if a future dataset is heavily imbalanced.

## 2. Train

```bash
python training/train_model.py
```

The model uses ImageNet-pretrained MobileNetV2 with a small head:

`GlobalAveragePooling2D -> Dense(128) -> Dropout(0.35) -> Dense(1, sigmoid)`

Training runs in two phases:

1. Frozen MobileNetV2 base.
2. Low-learning-rate fine-tuning of the top MobileNetV2 layers.

Expected free-tier GPU runtime for CIFAKE is roughly 45-90 minutes with default settings. Outputs:

- `training/output/best_model.h5`
- `training/output/final_model.h5`
- `training/output/training_log.csv`
- `training/output/history.json`

## 3. Convert to TensorFlow.js

```bash
python training/convert_to_tfjs.py
```

The converter writes an int8-quantized TF.js layers model to:

```text
training/output/tfjs_model/
```

The script prints total on-disk size and warns if it exceeds 10MB. MobileNetV2 with 1-byte weight quantization should typically land in the target browser-extension range.

## 4. Evaluate

Evaluate the original Keras checkpoint:

```bash
python training/evaluate.py
```

Optionally evaluate a TF.js round-trip conversion:

```bash
python training/evaluate.py --use-tfjs
```

Outputs:

- Confusion matrix in the terminal.
- `training/output/metrics.json`.
- Warning if test accuracy is below the 85% CIFAKE sanity target.

## Colab/Kaggle Notes

- Enable GPU: Runtime → Change runtime type → GPU in Colab, or Accelerator → GPU in Kaggle.
- Keep batch size at `64` for T4/P100-class GPUs; reduce to `32` if memory is tight.
- To speed up iteration, first smoke-test with `--initial-epochs 1 --fine-tune-epochs 1`.
- For better accuracy, increase epochs before increasing model size.

## End-to-End Commands

```bash
pip install -r training/requirements.txt
python training/prepare_data.py
python training/train_model.py
python training/convert_to_tfjs.py
python training/evaluate.py
```

Do not start Chrome extension integration until `training/output/tfjs_model/` exists and `training/output/metrics.json` reports acceptable test accuracy.
