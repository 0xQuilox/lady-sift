import argparse
import json
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

    # Patch Keras 3 InputLayer serialization for TF.js 4.x (batch_shape/optional -> batchInputShape)
    model_json = args.output_dir / "model.json"
    if model_json.exists():
        with model_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def patch(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") == "InputLayer" and "config" in obj:
                    cfg = obj["config"]
                    if "batch_shape" in cfg:
                        cfg["batchInputShape"] = cfg.pop("batch_shape")
                    cfg.pop("optional", None)
                for v in obj.values():
                    patch(v)
            elif isinstance(obj, list):
                for x in obj:
                    patch(x)

        patch(data)
        # Strip augmentation Sequential (RandomFlip/Rotation/Zoom not supported in TF.js)
        # and fix Keras 3 inbound_nodes object format -> array format for TF.js 4.x (recursive)
        try:
            layers = data["modelTopology"]["model_config"]["config"]["layers"]
            new_layers = [l for l in layers if l.get("name") != "augmentation"]
            if len(new_layers) != len(layers):
                for l in new_layers:
                    if l.get("name") == "rescaling":
                        for node in l.get("inbound_nodes", []):
                            if isinstance(node, dict) and "args" in node:
                                for arg in node.get("args", []):
                                    # handle both [dict] and [[dict, dict]] for Add
                                    tensors = arg if isinstance(arg, list) else [arg]
                                    if isinstance(arg, list):
                                        tensors = arg
                                    else:
                                        # single tensor case already handled, but keep
                                        pass
                                    for t in tensors if isinstance(tensors[0], list) else [arg] if isinstance(arg, dict) else []:
                                        hist = t.get("config", {}).get("keras_history") if isinstance(t, dict) else None
                                        if hist and hist[0] == "augmentation":
                                            hist[0] = "input_layer"
                            elif isinstance(node, list):
                                for entry in node:
                                    if isinstance(entry, list) and entry[0] == "augmentation":
                                        entry[0] = "input_layer"
                                    elif entry == "augmentation":
                                        entry = "input_layer"
                data["modelTopology"]["model_config"]["config"]["layers"] = new_layers
                print("Stripped augmentation layer for TF.js (RandomFlip/Rotation/Zoom)")

            def fix_inbound(layer):
                inns = layer.get("inbound_nodes")
                if not inns or not isinstance(inns, list):
                    return
                # check if already array format (first element is list of [str, int, int, dict])
                if inns and isinstance(inns[0], list) and len(inns[0]) > 0 and isinstance(inns[0][0], list) and isinstance(inns[0][0][0], str):
                    return
                if inns and isinstance(inns[0], dict):
                    new_inns = []
                    for node in inns:
                        if isinstance(node, dict) and "args" in node:
                            args = node.get("args", [])
                            kwargs = node.get("kwargs", {})
                            # Handle multi-input: args may be [[t1,t2]] or [t1]
                            tensors = []
                            if len(args) == 1 and isinstance(args[0], list):
                                tensors = args[0]
                            else:
                                tensors = args
                            new_args = []
                            for t in tensors:
                                if isinstance(t, dict):
                                    hist = t.get("config", {}).get("keras_history")
                                    if hist:
                                        new_args.append([hist[0], hist[1], hist[2], kwargs])
                            new_inns.append(new_args)
                        else:
                            new_inns.append(node)
                    if new_inns:
                        layer["inbound_nodes"] = new_inns

            def recurse_layers(obj):
                if isinstance(obj, dict):
                    if "class_name" in obj and "inbound_nodes" in obj:
                        fix_inbound(obj)
                    for v in obj.values():
                        recurse_layers(v)
                elif isinstance(obj, list):
                    for x in obj:
                        recurse_layers(x)

            recurse_layers(data)
            print("Fixed inbound_nodes object->array for TF.js")
        except Exception as e:
            print(f"Warning: failed to strip augmentation / fix inbound_nodes: {e}")
            import traceback; traceback.print_exc()

        with model_json.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        print("Patched model.json InputLayer for TF.js compatibility")

    size_bytes = directory_size(args.output_dir)
    print(f"TF.js model written to {args.output_dir}")
    print(f"TF.js model size: {size_bytes / (1024 * 1024):.2f} MB")
    if size_bytes > 10 * 1024 * 1024:
        print("WARNING: TF.js model exceeds the 10MB target.")


if __name__ == "__main__":
    main()
