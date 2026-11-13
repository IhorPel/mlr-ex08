"""Generate experiment evidence and notebook result notes, leaving student reports separate."""
import base64
import csv
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    with (ROOT / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evidence = ROOT / "evidence"
    selection = json.loads((evidence / "phone_selection.json").read_text(encoding="utf-8"))
    histories = json.loads((evidence / "histories.json").read_text(encoding="utf-8"))
    times = json.loads((evidence / "runtimes.json").read_text(encoding="utf-8"))
    task2 = json.loads((evidence / "task2.json").read_text(encoding="utf-8"))
    provenance = json.loads((evidence / "data_provenance.json").read_text(encoding="utf-8"))
    chosen = selection["variant"]
    optional = json.loads((evidence / "task1_optional.json").read_text(encoding="utf-8"))
    optional_rows = optional["split_and_dropout"]
    header = ["| Variant | Seed | Epoch | Clean validation | Camera-like validation |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        header.append(f"| {row['variant']} | {row['seed']} | {row['selected_epoch']} | {float(row['validation_accuracy']):.4f} | {float(row['camera_accuracy']):.4f} |")
    repeated = sorted({r["variant"] for r in rows if r["seed"] != "0"})
    seed_lines = ["| Repeated family | Seeds | Clean mean / SD | Camera-like mean / SD |", "| --- | --- | ---: | ---: |"]
    for name in repeated:
        values = [r for r in rows if r["variant"] == name]
        clean = np.array([float(r["validation_accuracy"]) for r in values])
        camera = np.array([float(r["camera_accuracy"]) for r in values])
        seed_lines.append(f"| {name} | {', '.join(r['seed'] for r in values)} | {clean.mean():.4f} / {clean.std(ddof=0):.4f} | {camera.mean():.4f} / {camera.std(ddof=0):.4f} |")
    baseline = next(r for r in rows if r["variant"] == "baseline" and r["seed"] == "0")
    task3_rows = [r for r in rows if r["variant"] in {"baseline", "geometry", "batchnorm", "bn_geometry"} and r["seed"] == "0"]
    task3_table = [header[0], header[1]] + [f"| {r['variant']} | 0 | {r['selected_epoch']} | {float(r['validation_accuracy']):.4f} | {float(r['camera_accuracy']):.4f} |" for r in task3_rows]
    cm = np.array(task2["clean_confusion"])
    classes = list(provenance["class_counts"]["val"])
    pairs = [(int(cm[i,j]+cm[j,i]), classes[i], classes[j]) for i in range(4) for j in range(i+1,4)]
    confusion_count, class1, class2 = max(pairs)
    doc = (
        "# Generated experiment evidence\n\n"
        "This file contains measured results to inspect. It is not the student's REVIEW.md or REPORT.md.\n\n"
        "## Dataset\n\nThe user supplied the historical archive instead of the current-group Drive ZIP. "
        "There are 175 training and 30 validation images per class, 700 and 120 total. "
        "These results do not establish performance on the current group's photographs. "
        "Camera-like validation contains one fixed crop, rotation and colour change of each of the "
        "same 120 validation images, with seed 123. It is a stress test, not an independent camera dataset.\n\n"
        "## Task 1 optional exercises\n\n"
        f"Batch size 1 with learning rate 0.01 gives final half-MSE {optional['batch1_eta001']['mean_half_mse']:.6f}. "
        f"Full-batch gradient descent at learning rate 0.1 reaches half-MSE below 0.0569 after {optional['fullbatch_eta01']['epochs']} epochs. "
        f"For 300 epochs, split seeds 0, 1 and 2 give best validation MSE {optional_rows[0]['best_validation_mse']:.4f}, "
        f"{optional_rows[1]['best_validation_mse']:.4f} and {optional_rows[2]['best_validation_mse']:.4f}, "
        f"at epochs {optional_rows[0]['best_epoch']}, {optional_rows[1]['best_epoch']} and {optional_rows[2]['best_epoch']}. "
        f"Dropout 0.2 on split seed 0 gives {optional_rows[3]['best_validation_mse']:.4f} at epoch {optional_rows[3]['best_epoch']}. "
        "The best epoch is no longer always the last, and a different split can change the result substantially. "
        "Dropout did not improve this particular split. Exact measured values are in evidence/task1_optional.json.\n\n"
        "## Task 2\n\n"
        f"The demonstration's clean accuracy is {task2['validation_accuracy']:.4f}, and its camera-like accuracy "
        f"is {task2['camera_accuracy']:.4f}. The most-confused species pair is {class1}/{class2}, with "
        f"{confusion_count} errors across both directions. The saved ONNX check records agreement with PyTorch. "
        "Browser parity and a real phone check are separate verification steps.\n\n"
        "## Task 3\n\n" + "\n".join(task3_table) + "\n\n"
        "## Task 4: all measured variants\n\n" + "\n".join(header) + "\n\n"
        "The CNN baseline is the notebook's four-block model trained for 15 epochs with Adam at 0.001. "
        "Single-change rows retain that baseline's settings. Frozen/fine-tuned MobileNet is a transfer-learning "
        "comparison; transfer geometry and transfer cosine are each one change relative to fine-tuned MobileNet. "
        "Combination rows change multiple settings and do not isolate their individual effects. "
        "The fine-tuned transfer seed-0 result reuses the identical Task 2 model, and its recorded training time is zero to make the reuse explicit.\n\n"
        "## Seed variability\n\n" + "\n".join(seed_lines) + "\n\n"
        f"The exported Task 4 phone candidate is **{chosen}**, using seed 0. The family was chosen by mean "
        "camera-like accuracy between the repeated baseline and the initially best candidate. Seed 0 is used "
        "for export without selecting the best random seed. Other variants have only one seed, so their ranking "
        "remains exploratory. Standard deviations across three seeds describe initialization/training variability; "
        "they are not confidence intervals or a formal significance test.\n\n"
        "One validation image changes accuracy by 1/120 = **0.833 percentage points**. The two validation "
        "sets share the same source images, so they are not independent measurements. Repeatedly selecting "
        "variants and epochs on them creates validation-selection optimism. No untouched final test set is available.\n\n"
        "## Runtime\n\n"
        f"Recorded training wall time across actual runs: {sum(times.values()):.1f} seconds. Per-run values "
        "are in evidence/runtimes.json. They include interruptions and machine load, so they should not be "
        "treated as hardware benchmarks. A zero value denotes reuse, not a measured zero-cost training run.\n\n"
        "## Remaining student checks\n\n"
        "Open the deployed URL on an actual Android phone or iPhone, test the rear camera, and take the "
        "required screenshot yourself. Confirm check_image.jpg probabilities in the browser. Try the Task 4 "
        "candidate in the course camera app if required. Write REVIEW.md and REPORT.md in your own words "
        "after checking the code and judging the results. A desktop-browser test cannot establish physical iPhone camera behavior.\n")
    (ROOT / "EXPERIMENTS.md").write_text(doc, encoding="utf-8")
    # Scientific result plots: no claim of extra independent validation data.
    seed0 = [r for r in rows if r["seed"] == "0"]
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(seed0)*.34)))
    y = np.arange(len(seed0))
    ax.barh(y-.16, [float(r["validation_accuracy"]) for r in seed0], height=.3, label="Clean validation", color="#215d46")
    ax.barh(y+.16, [float(r["camera_accuracy"]) for r in seed0], height=.3, label="Camera-like validation", color="#acb978")
    ax.set(yticks=y, yticklabels=[r["variant"] for r in seed0], xlabel="Accuracy", xlim=(0,1), title="Same 120 validation images: clean and fixed perturbations (seed 0)")
    ax.invert_yaxis(); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ROOT / "comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(11,4.2))
    for name in dict.fromkeys(["baseline", "bn_geometry", "demo_mobilenet", chosen]):
        history = histories[f"{name}_seed0"]
        axes[0].plot([r["epoch"] for r in history], [r["training_loss"] for r in history], label=name)
        axes[1].plot([r["epoch"] for r in history], [r["validation_accuracy"] for r in history], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Mean training cross-entropy", yscale="log")
    axes[1].set(xlabel="Epoch", ylabel="Clean validation accuracy", ylim=(0,1))
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / "learning_curves.png", dpi=150); plt.close(fig)
    def img(name): return "data:image/png;base64," + base64.b64encode((ROOT/name).read_bytes()).decode()
    table_rows = "".join(f"<tr><td>{html.escape(r['variant'])}</td><td>{r['seed']}</td><td>{float(r['validation_accuracy']):.4f}</td><td>{float(r['camera_accuracy']):.4f}</td></tr>" for r in rows)
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exercise 8 measured evidence</title><style>body{{font:16px/1.6 system-ui;background:#f3f5ed;color:#153f33;margin:0}}main{{max-width:1100px;margin:25px auto;background:white;padding:30px;border-radius:14px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #d9e1d5;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}}</style><main>
<h1>Exercise 8: measured experiment evidence</h1><p>Historical data supplied by the user. Current-group photos and a physical phone were not tested by the agent.</p>
<img src="{img('comparison.png')}" alt="Clean and camera-like validation accuracy for all seed-zero variants">
<img src="{img('learning_curves.png')}" alt="Training loss and validation learning curves">
<table><tr><th>Variant</th><th>Seed</th><th>Clean accuracy</th><th>Camera-like accuracy</th></tr>{table_rows}</table>
<h2>Methods, limitations and student checks</h2><pre>{html.escape(doc)}</pre></main></html>"""
    (ROOT / "experiment_report.html").write_text(body, encoding="utf-8")
    nb_path = ROOT / "ex08_notebook.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"][47]["source"] = [f"On the supplied historical validation split, {class1} and {class2} form the most confused pair ({confusion_count} errors in both directions). Similar leaf appearance, cropping, lighting or background may contribute; the confusion matrix alone does not establish the cause.\n"]
    bn_combo = next(r for r in task3_rows if r["variant"] == "bn_geometry")
    nb["cells"][62]["source"] = [f"With seed 0, the baseline gives {float(baseline['validation_accuracy']):.4f} clean accuracy and {float(baseline['camera_accuracy']):.4f} camera-like accuracy. Augmentation plus batch normalization gives {float(bn_combo['validation_accuracy']):.4f} and {float(bn_combo['camera_accuracy']):.4f}. The changes do not improve both measures together. One validation image is worth 0.833 percentage points. These single-seed comparisons are exploratory; see the repeated important rows and their variability in EXPERIMENTS.md.\n"]
    nb["cells"] = [c for c in nb["cells"] if not c.get("metadata", {}).get("measured_results")]
    nb["cells"].append({"cell_type":"markdown", "id":"measured-results", "metadata":{"measured_results":True},
                         "source":doc.splitlines(keepends=True)})
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(f"Saved evidence, plots and notebook results for {len(rows)} measured rows. Selected: {chosen}")


if __name__ == "__main__": main()
