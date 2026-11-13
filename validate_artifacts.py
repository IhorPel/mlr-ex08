"""Check exported assets, saved metrics and publication contents without retraining."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent
info = json.loads((ROOT / "model_info.json").read_text(encoding="utf-8"))
assert info == {"classes": ["apple", "cherry", "chestnut", "maple"], "input_size": 96,
                "mean": [.485, .456, .406], "std": [.229, .224, .225]}
for name in ["leaf_model.onnx", "best_leaf_model.onnx"]:
    model = onnx.load(ROOT / name)
    onnx.checker.check_model(model)
    metadata = {p.key: p.value for p in model.metadata_props}
    assert json.loads(metadata["model_info"]) == info
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [1,3,96,96]
for name in ["demo_export_check.json", "best_export_check.json"]:
    check = json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))
    assert check["maximum_probability_difference"] < 1e-4
with (ROOT / "results.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert all(int(r["train_images"]) == 700 and int(r["validation_images"]) == 120 for r in rows)
for row in rows:
    for key in ["validation_accuracy", "camera_accuracy"]:
        count = float(row[key]) * 120
        assert abs(count-round(count)) < 1e-10
selection = json.loads((ROOT / "evidence/phone_selection.json").read_text(encoding="utf-8"))
for name in {"baseline", selection["variant"]}:
    assert {int(r["seed"]) for r in rows if r["variant"] == name} == {0,1,2}
nb = json.loads((ROOT / "ex08_notebook.ipynb").read_text(encoding="utf-8"))
for index in [54,56,58,60,61]:
    source = "".join(nb["cells"][index]["source"])
    assert "PLACEHOLDER" not in source and "TODO" not in source
task1 = json.loads((ROOT / "evidence/task1.json").read_text(encoding="utf-8"))
assert task1["saved_and_loaded_predictions_equal"]
assert abs(task1["single_sample_final_half_mse"] - .0568768329109) < 1e-10
qr = json.loads((ROOT / "evidence/qr_check.json").read_text(encoding="utf-8"))
assert qr["encoded_url"] == "https://ihorpel.github.io/mlr-ex08/"
source = (ROOT / "index.html").read_text(encoding="utf-8")
assert qr["encoded_url"] in source and "numThreads = 1" in source
result = {"passed": True, "measured_rows": len(rows), "same_training_and_validation_counts": True,
          "three_seeds_for_important_rows": True, "onnx_metadata_matches_json": True,
          "native_onnx_matches_pytorch": True, "task3_placeholders_replaced": True,
          "physical_phone_test": "Requires student verification; not performed by the agent"}
(ROOT / "evidence/artifact_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
