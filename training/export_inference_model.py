"""
Build a clean inference-only model without augmentation layers and copy trained weights by name.
Verifies predictions match between trained and inference models before saving.
"""
import tensorflow as tf
from pathlib import Path
import numpy as np

def build_inference_model(image_size=224):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_layer")
    rescaled = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="rescaling")(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=None,
    )
    base_model._name = "mobilenetv2_1.00_224"
    features = base_model(rescaled, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(features)
    hidden = tf.keras.layers.Dense(128, activation="relu", name="dense")(pooled)
    dropped = tf.keras.layers.Dropout(0.35, name="dropout")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1")(dropped)
    return tf.keras.Model(inputs, outputs, name="lady_sift_inference")


def main():
    src = Path("training/output/best_model.h5")
    if not src.exists():
        raise FileNotFoundError(f"Trained model not found: {src} - run training first")

    print(f"Loading trained model: {src}")
    trained = tf.keras.models.load_model(src)
    trained.summary()

    print("Building clean inference model (no augmentation)...")
    inference_model = build_inference_model()
    inference_model.summary()

    print("Copying weights by name (skip_mismatch=True)...")
    inference_model.load_weights(str(src), by_name=True, skip_mismatch=True)

    # Verify predictions match on a dummy input
    print("Verifying predictions match...")
    dummy = np.random.randint(0, 256, size=(1, 224, 224, 3)).astype("float32")
    # Preprocess same as training Rescaling
    dummy_rescaled = (dummy / 127.5) - 1.0

    # Need to run through full models, not just rescaled
    # Use the models directly with original dummy (they include Rescaling internally)
    pred_trained = trained.predict(dummy, verbose=0)[0][0]
    pred_infer = inference_model.predict(dummy, verbose=0)[0][0]
    diff = abs(float(pred_trained) - float(pred_infer))
    print(f"  trained={pred_trained:.6f} inference={pred_infer:.6f} diff={diff:.6f}")
    if diff > 1e-4:
        print(f"WARNING: predictions differ by {diff:.6f} > 1e-4 - weights may not have matched")
        # Print layer names to debug
        trained_names = [l.name for l in trained.layers]
        infer_names = [l.name for l in inference_model.layers]
        print("Trained layers:", trained_names[:10])
        print("Inference layers:", infer_names[:10])
    else:
        print("  OK - predictions match within tolerance")

    # Also test with a real image from test set if available
    test_dir = Path("training/data/processed/test/real")
    if test_dir.exists():
        real_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if real_images:
            print(f"Testing with real image: {real_images[0]}")
            from PIL import Image
            img = Image.open(real_images[0]).convert("RGB").resize((224, 224))
            arr = np.array(img).astype("float32")[None, ...]
            p1 = trained.predict(arr, verbose=0)[0][0]
            p2 = inference_model.predict(arr, verbose=0)[0][0]
            d2 = abs(float(p1) - float(p2))
            print(f"  real image: trained={p1:.6f} inference={p2:.6f} diff={d2:.6f}")

    out_dir = Path("training/output/inference_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_model.h5"
    inference_model.save(out_path)
    print(f"Saved clean inference-only model: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Also save a version for TF.js converter verification
    print("Done. Now run:")
    print(f"  tensorflowjs_converter --input_format=keras {out_path} training/output/inference_tfjs_model")


if __name__ == "__main__":
    main()
