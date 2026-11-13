# Exercise 8: Leaf Lens

Phone page: **https://ihorpel.github.io/mlr-ex08/**

This public repository contains the completed PyTorch notebook code, the Task 2
MobileNet export and browser classifier, and the measured Task 3/4 experiments.
The page runs inference locally with ONNX Runtime Web. It has a rear-camera view,
photo upload fallback, probabilities for all four classes, a notebook-parity check,
and a locally generated QR code pointing to this app's URL.

## Dataset choice

The notebook's GROUP_ZIP was empty. The user supplied the **historical** course ZIP:

https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip

It contains 175 training and 30 validation images per species: 700 training and
120 validation images in total. Classes are sorted as apple, cherry, chestnut,
maple. This is a deliberate user-supplied dataset substitution. **These are not
results on the current group's Drive ZIP**, and they do not establish performance
on that group's photographs. Data provenance and the ZIP hash are recorded in
`evidence/data_provenance.json`. The raw dataset is not committed.

## Task by task

Task 1's supplied demonstrations were executed on CPU, with captured notebook
outputs. The optional batch-size, split-seed and dropout measurements are saved
in `evidence/task1_optional.json`. Saved/reloaded predictions match exactly.

Task 2 trains MobileNet V3 Small for 15 epochs, exports a single-file opset-17
ONNX model, embeds preprocessing/class metadata, and saves the four requested
files. The demo keeps the final epoch, matching the notebook. Native ONNX Runtime
probabilities are checked against PyTorch in `evidence/demo_export_check.json`.
The live browser parity check is a separate verification. ONNX Runtime Web 1.23.2
is loaded from a pinned jsDelivr URL and uses one WASM thread for Safari support.

Task 3's placeholders are replaced with a readable Dataset, four-convolution
network, explicit training loop, geometric augmentation and batch normalization.
The Dataset is tested against ImageFolder. The notebook's plain CNN and the
experiment baseline have identical initial weights and predictions.

Task 4 measures one change at a time against that CNN baseline, then combinations.
Frozen and fine-tuned MobileNet form a separate transfer comparison; geometry and
cosine scheduling are each also measured relative to fine-tuned MobileNet. The
strongest seed-0 candidate and the scratch baseline are repeated with seeds 1 and
2. The Task 4 export uses seed 0 of the selected family, not the best random seed.
The page's model selector lets you compare the original Task 2 demo with that
Task 4 candidate.

`EXPERIMENTS.md`, `experiment_report.html`, `results.csv`, `comparison.png` and
`learning_curves.png` contain the measured evidence. `REVIEW.md` and `REPORT.md`
are intentionally left for the student's own device check and interpretation.
Do not treat the generated evidence as a claim that a physical phone was tested.

## Fresh-clone commands on Windows

Prerequisites: Git, uv, Python 3.14 (uv can provision it) and network access for
packages, pretrained weights and the dataset. The lock file pins the tested CPU
runtime. The notebook can separately use CUDA in Colab; GPU timings were not
measured in this run.

```powershell
git clone https://github.com/IhorPel/mlr-ex08.git
cd mlr-ex08
uv sync --locked --system-certs
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip' -OutFile leaves.zip
uv run --locked python prepare_data.py leaves.zip
uv run --locked pytest -v -p no:cacheprovider
uv run --locked python run_notebook.py 1
uv run --locked python task1_optional.py
uv run --locked python complete_notebook.py
uv run --locked python experiments.py all
uv run --locked python summarize_results.py
uv run --locked python make_qr.py
uv run --locked python validate_artifacts.py
```

`experiments.py demo` runs just Task 2, and `experiments.py compare` runs the
Task 3/4 comparisons. Completed rows are reused only when both their evidence JSON
and local checkpoint exist. Results are saved after every run so an interruption
can be resumed. Checkpoints and the raw dataset are excluded from Git. Existing
evidence without its local checkpoint is retrained from the fixed seed.

The CPU experiment uses four native PyTorch threads, `torch.use_deterministic_algorithms(True)`,
and seeds 0, 1 and 2 as recorded in each row. All standard runs use 15 epochs,
batch size 32 and Adam with learning rate 0.001. The explicit transfer combination
uses 30 epochs, geometry, colour, cosine scheduling and a best-validation checkpoint.
Frozen transfer keeps feature weights, batch-normalization statistics and frozen
dropout behavior fixed. For the CNN, each switch is limited to its corresponding
change: batch normalization, geometry, colour, dropout 0.3, global pooling, AdamW
with weight decay 0.001, label smoothing 0.1, best epoch or cosine scheduling.

Both clean and camera-like validation are measured for every variant. The latter
contains one saved crop, rotation and colour perturbation per validation image,
generated once with seed 123. These are derived views of the same images, not an
independent test set. One validation image changes accuracy by 0.833 percentage
points. Three-seed SD is descriptive variability, not a significance test.

Per-run measured wall times are in `evidence/runtimes.json`. These include machine
load and interruptions; they should not be used as hardware benchmarks. Zero for
the transfer_fine seed-0 row denotes explicit reuse of the already-trained Task 2
model, not a claim of zero-cost training.

## What the browser does

`index.html` crops the central square, resizes it to the input size from
`model_info.json`, divides RGB bytes by 255, normalizes channels by mean/std and
packs a `[1,3,96,96]` float32 tensor. Softmax is applied to model logits in the
browser. The model receives the same channel order and normalization as PyTorch.

On a phone, camera access requires HTTPS and user permission. The video is muted
and uses playsinline for Safari. Frames are classified sequentially with no
overlapping inference requests. Stop, leaving the page or hiding it releases the
camera. Photo upload and the built-in check image use the same tensor processing.

No camera frame or selected photo is uploaded by this app. Network requests fetch
this site's model, metadata, check image and QR, plus JavaScript/WebAssembly from
the pinned ONNX Runtime Web CDN. No external QR-image service is used. QR payload
and rendered-module verification is saved in `evidence/qr_check.json`.

## Finish the student hand-in

Open the page on your actual phone, test a leaf with the rear camera, and capture
the screenshot required by Task 2. Use Run check image or upload check_image.jpg
to compare against check_expected.json. Check the preprocessing and network
requests yourself, then write REVIEW.md in your own words.

Inspect the measured table, what helped or failed, the seed variability and the
limitations. Try the Task 4 export in the course camera app if required, then write
your own REPORT.md with your judgment about which model to put on the phone.
Those two student-authored files are still required for the compulsory hand-in.
Keep personal photos, personal information, raw data and secrets out of this
public repository. MLRteacher must have access or a pending collaborator invitation.
