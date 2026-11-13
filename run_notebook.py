"""Execute the notebook's supplied demonstrations with captured figures and output."""
import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def execute_cells(notebook, indices, namespace=None):
    namespace = namespace or {"__name__": "__notebook__"}
    for i in indices:
        source = "".join(notebook["cells"][i]["source"])
        # Dependencies are installed through the lock file, not inside notebook execution.
        source = "\n".join(line[:len(line)-len(line.lstrip())] + "pass # dependencies installed by uv"
                           if line.lstrip().startswith("%") else line for line in source.splitlines())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"notebook-cell-{i}", "exec"), namespace)
        outputs = []
        if buffer.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": buffer.getvalue().splitlines(keepends=True)})
            print(f"Cell {i}: {buffer.getvalue().strip()}", flush=True)
        for figure_number in plt.get_fignums():
            fig = plt.figure(figure_number)
            data = io.BytesIO()
            fig.savefig(data, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(data.getvalue()).decode("ascii")}})
            plt.close(fig)
        notebook["cells"][i]["outputs"] = outputs
        notebook["cells"][i]["execution_count"] = i + 1
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["1"])
    parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    plt.show = lambda *args, **kwargs: None
    path = ROOT / "ex08_notebook.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    indices = [i for i in range(35) if nb["cells"][i]["cell_type"] == "code"]
    namespace = execute_cells(nb, indices)
    evidence = {"device": str(namespace["device"]), "torch": torch.__version__,
                "single_sample_final_half_mse": float(namespace["curve_0"][-1]),
                "batched_final_half_mse": float(namespace["curve_1"][-1]),
                "module_final_half_mse": float(namespace["curve_2"][-1]),
                "sgd_final_mse": float(namespace["curves_3"]["SGD"][-1]),
                "adam_final_mse": float(namespace["curves_3"]["Adam"][-1]),
                "saved_and_loaded_predictions_equal": bool(namespace["same"]),
                "best_validation_mse": float(namespace["best_loss"]),
                "best_validation_epoch": int(namespace["best_epoch"]),
                "note": "CPU-only environment; GPU demonstrations are conditional and were not measured"}
    assert evidence["saved_and_loaded_predictions_equal"]
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/task1.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
