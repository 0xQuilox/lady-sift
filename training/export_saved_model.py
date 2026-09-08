"""
Export clean inference model as SavedModel for GraphModel conversion.
Reuses training/output/inference_model/inference_model.h5 if exists, otherwise builds fresh.
"""
import tensorflow as tf
from pathlib import Path

def main():
    src = Path("training/output/inference_model/inference_model.h5")
    if not src.exists():
        # fallback to best_model.h5 via export_inference_model
        print("inference_model.h5 not found, building fresh inference model...")
        from export_inference_model import build_inference_model
        trained = tf.keras.models.load_model("training/output/best_model.h5")
        model = build_inference_model()
        for layer in model.layers:
            try:
                src_layer = trained.get_layer(layer.name)
                w = src_layer.get_weights()
                if w:
                    layer.set_weights(w)
            except:
                pass
    else:
        print(f"Loading inference model: {src}")
        model = tf.keras.models.load_model(src)

    out = Path("training/output/saved_model")
    # Remove old saved_model contents if exists
    import shutil
    if out.exists():
        shutil.rmtree(out)

    # Keras 3: use model.export for SavedModel
    try:
        model.export(str(out))
    except AttributeError:
        tf.saved_model.save(model, str(out))
    print(f"Saved SavedModel to {out}")
    # Verify
    loaded = tf.saved_model.load(str(out))
    print(f"SavedModel signatures: {list(loaded.signatures.keys())}")

if __name__ == "__main__":
    main()
