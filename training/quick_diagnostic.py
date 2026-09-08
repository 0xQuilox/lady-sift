"""
Diagnostic: check if model scores spread or cluster near 0 on real Twitter vs modern AI images.
Place 5-10 real Twitter photos in diagnostic_data/real/ and 5-10 Midjourney/SDXL images in diagnostic_data/fake/
"""
import tensorflow as tf
from pathlib import Path

model = tf.keras.models.load_model("training/output/inference_model/inference_model.h5")
print(f"Loaded {model.name}, input {model.input_shape}")

# Gather test images: diagnostic_data/real/* and diagnostic_data/fake/*
base = Path("diagnostic_data")
test_images = {}
for label in ["real", "fake"]:
    d = base / label
    if d.exists():
        for p in d.glob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                test_images[p] = label

if not test_images:
    print("No diagnostic_data/real or fake images found. Add 5-10 each, then re-run.")
    print("Example: mkdir -p diagnostic_data/real diagnostic_data/fake")
    exit(0)

for path, true_label in sorted(test_images.items()):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None, ...]
    score = model.predict(arr, verbose=0)[0][0]
    print(f"{path.name:40} score={score:.4f} true={true_label} pred={'fake' if score>=0.5 else 'real'}")

# Summary
scores = []
for path, _ in test_images.items():
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None, ...]
    scores.append(model.predict(arr, verbose=0)[0][0])
print(f"\nScore range: min={min(scores):.4f} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}")
if max(scores) - min(scores) < 0.1:
    print("DIAGNOSIS: scores cluster near 0 -> domain shift, need GenImage/ArtiFact retrain")
else:
    print("DIAGNOSIS: scores spread -> calibration/threshold may help")
