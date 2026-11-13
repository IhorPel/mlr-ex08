"""Run the demonstration, simple one-change comparisons, combinations and seed repeats."""
import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from leaf_pipeline import LeafDataset, MEAN, STD, image_transform, make_model, train_variant


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def export_model(checkpoint, *, best=False):
    import onnx
    import onnxruntime as ort
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model(saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval().cpu()
    name = "best_leaf_model.onnx" if best else "leaf_model.onnx"
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), str(ROOT / name),
                      input_names=["image"], output_names=["logits"], external_data=False,
                      dynamo=False, opset_version=17)
    info = {"classes": saved["classes"], "input_size": 96, "mean": MEAN, "std": STD}
    graph = onnx.load(ROOT / name)
    graph.metadata_props.add(key="model_info", value=json.dumps(info))
    onnx.checker.check_model(graph)
    onnx.save(graph, ROOT / name)
    validation = LeafDataset(ROOT / "data/leaves/val", image_transform())
    if not (ROOT / "check_image.jpg").exists():
        with Image.open(validation.samples[0][0]) as original:
            original.convert("RGB").save(ROOT / "check_image.jpg", quality=95)
    with Image.open(ROOT / "check_image.jpg") as original:
        x = image_transform()(original.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(x).softmax(dim=1)[0].numpy()
    session = ort.InferenceSession(str(ROOT / name), providers=["CPUExecutionProvider"])
    logits = session.run(None, {"image": x.numpy()})[0][0]
    onnx_probabilities = np.exp(logits - logits.max())
    onnx_probabilities /= onnx_probabilities.sum()
    difference = float(np.max(np.abs(onnx_probabilities - probabilities)))
    if difference > 1e-4:
        raise AssertionError(f"ONNX/PyTorch mismatch: {difference}")
    expected = dict(zip(saved["classes"], [float(p) for p in probabilities]))
    write_json(ROOT / ("best_check_expected.json" if best else "check_expected.json"), expected)
    write_json(ROOT / "model_info.json", info)
    write_json(ROOT / "evidence" / ("best_export_check.json" if best else "demo_export_check.json"),
               {"maximum_probability_difference": difference, "torch": expected,
                "onnx": dict(zip(saved["classes"], [float(p) for p in onnx_probabilities])),
                "model": name, "checkpoint_variant": saved["config"], "epoch": saved["epoch"], "seed": saved["seed"]})
    print(f"Exported {name}; ONNX/PyTorch maximum probability difference {difference:.3e}", flush=True)
    if not best:
        with zipfile.ZipFile(ROOT / "ex08_for_agent.zip", "w") as archive:
            for file in [name, "model_info.json", "check_image.jpg", "check_expected.json"]:
                archive.write(ROOT / file, file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["demo", "compare", "all"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    rows = []
    history_path = evidence / "histories.json"
    histories = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    times_path = evidence / "runtimes.json"
    times = json.loads(times_path.read_text(encoding="utf-8")) if times_path.exists() else {}
    def run(config, seed=0, epochs=15, phase="single_change"):
        key = f"{config['name']}_seed{seed}"
        checkpoint = ROOT / "checkpoints" / f"{key}.pt"
        result_path = evidence / f"{key}.json"
        if result_path.exists() and checkpoint.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["config"] != config or result["epochs"] != epochs or result["seed"] != seed:
                raise ValueError(f"Cached configuration differs for {key}; preserve it and use a fresh run folder")
            print(f"Reusing completed {key}", flush=True)
        else:
            started = time.perf_counter()
            result, history = train_variant(config, seed, ROOT / "data/leaves/train", ROOT / "data/leaves/val",
                                            ROOT / "data/camera_val", checkpoint, epochs=epochs)
            result["phase"] = phase
            result["config"] = config
            histories[key] = history
            times[key] = time.perf_counter() - started
            write_json(result_path, result)
            write_json(history_path, histories)
            write_json(times_path, times)
        rows.append(result)
        # Save every completed row, even if the rest of the run is interrupted.
        with (ROOT / "results.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["variant", "seed", "phase", "epochs", "selected_epoch", "validation_accuracy", "camera_accuracy",
                      "validation_balanced_accuracy", "camera_balanced_accuracy", "train_images", "validation_images", "parameters", "config"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json.dumps(row[name], sort_keys=True) if name == "config" else row[name] for name in fields})
        return result, checkpoint

    if args.phase in {"demo", "all"}:
        demo, checkpoint = run({"name": "demo_mobilenet", "transfer": "fine"}, phase="task2_demo")
        export_model(checkpoint)
        write_json(evidence / "task2.json", demo)
    if args.phase in {"compare", "all"}:
        configs = [
            {"name": "baseline"},
            {"name": "geometry", "geometric": True},
            {"name": "batchnorm", "batch_norm": True},
            {"name": "bn_geometry", "batch_norm": True, "geometric": True},
            {"name": "colour", "colour": True},
            {"name": "dropout", "dropout": .3},
            {"name": "global_pool", "global_pool": True},
            {"name": "adamw", "adamw": True},
            {"name": "label_smoothing", "label_smoothing": .1},
            {"name": "best_epoch", "best_epoch": True},
            {"name": "cosine", "scheduler": True},
            {"name": "transfer_frozen", "transfer": "frozen"},
            {"name": "transfer_fine", "transfer": "fine"},
            {"name": "transfer_geometry", "transfer": "fine", "geometric": True},
            {"name": "transfer_cosine", "transfer": "fine", "scheduler": True},
        ]
        for config in configs:
            phase = "baseline" if config["name"] == "baseline" else "task3_combination" if config["name"] == "bn_geometry" else "transfer_comparison" if config.get("transfer") else "single_change"
            # This is exactly the trained demo, with the same architecture/settings/seed.
            if config["name"] == "transfer_fine" and (evidence / "demo_mobilenet_seed0.json").exists():
                demo_result = json.loads((evidence / "demo_mobilenet_seed0.json").read_text(encoding="utf-8"))
                copy_result = dict(demo_result, variant="transfer_fine", config=config, phase=phase)
                source = ROOT / "checkpoints/demo_mobilenet_seed0.pt"
                saved = torch.load(source, map_location="cpu", weights_only=True)
                saved["config"] = config
                torch.save(saved, ROOT / "checkpoints/transfer_fine_seed0.pt")
                write_json(evidence / "transfer_fine_seed0.json", copy_result)
                histories["transfer_fine_seed0"] = histories["demo_mobilenet_seed0"]
                times["transfer_fine_seed0"] = 0.0  # Explicit reuse, not a second training claim.
                write_json(history_path, histories)
                write_json(times_path, times)
            run(config, phase=phase)
        # Select two beneficial single changes for a measured scratch-network combination.
        baseline = next(row for row in rows if row["variant"] == "baseline")
        singles = [row for row in rows if row["phase"] == "single_change" and row["camera_accuracy"] > baseline["camera_accuracy"]]
        singles.sort(key=lambda r: (r["camera_accuracy"], r["validation_accuracy"]), reverse=True)
        combination = {"name": "selected_scratch_combination"}
        for row in singles[:2]:
            combination.update({k: v for k, v in row["config"].items() if k != "name"})
        if len(combination) > 1:
            run(combination, phase="combination")
        run({"name": "transfer_aug_cosine_30", "transfer": "fine", "geometric": True,
             "colour": True, "scheduler": True, "best_epoch": True}, epochs=30, phase="combination")
        eligible = [row for row in rows if row["variant"] != "demo_mobilenet"]
        candidate = max(eligible, key=lambda r: (r["camera_accuracy"], r["validation_accuracy"]))
        repeated = [next(r for r in rows if r["variant"] == "baseline"), candidate]
        for row in repeated:
            for seed in [1, 2]:
                if any(r["variant"] == row["variant"] and r["seed"] == seed for r in rows):
                    continue
                run(row["config"], seed=seed, epochs=row["epochs"], phase="seed_repeat")
        families = {row["variant"] for row in repeated}
        means = {name: float(np.mean([r["camera_accuracy"] for r in rows if r["variant"] == name])) for name in families}
        chosen = max(sorted(families), key=lambda name: means[name])
        write_json(evidence / "phone_selection.json", {"variant": chosen, "checkpoint_seed": 0,
                   "rule": "Choose between the repeated baseline and seed-0 winning candidate by mean camera-like validation accuracy; export seed 0 without choosing the best seed",
                   "camera_means": means, "data_source": "historical archive supplied by user",
                   "physical_phone_test": "Not performed by the agent; requires the user's device"})
        export_model(ROOT / "checkpoints" / f"{chosen}_seed0.pt", best=True)
        print(f"Phone candidate: {chosen}; seed means {means}", flush=True)


if __name__ == "__main__":
    main()
