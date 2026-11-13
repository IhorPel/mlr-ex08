"""Fill the readable Task 3 cells and record the supplied data source explicitly."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / "ex08_notebook.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))


def replace(index, source):
    nb["cells"][index]["source"] = source.strip().splitlines(keepends=True)
    nb["cells"][index]["outputs"] = []
    nb["cells"][index]["execution_count"] = None


replace(38, '''# The user supplied the historical course ZIP instead of a current-group Drive ZIP.
GROUP_ZIP = "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip"

from pathlib import Path
from urllib.parse import urlparse
import gdown

GROUP_ZIP = GROUP_ZIP.strip()
if not GROUP_ZIP:
    raise ValueError("Provide the teacher's shared ZIP link first.")
link = urlparse(GROUP_ZIP)
if link.hostname == "drive.google.com" and "/folders/" in link.path:
    raise ValueError("Use a Drive file link, not a folder link.")
if link.hostname not in {"drive.google.com", "raw.githubusercontent.com"}:
    raise ValueError("Use a Drive file link or the supplied raw course archive URL.")

os.makedirs("data", exist_ok=True)
source_file = Path("data/leaves/.group_zip_url")
if Path("data/leaves").exists():
    if not source_file.exists() or source_file.read_text().strip() != GROUP_ZIP:
        raise ValueError("Cached data use another source. Preserve them and use a new project folder.")
else:
    if link.hostname == "drive.google.com":
        downloaded = gdown.download(url=GROUP_ZIP, output="data/leaves.zip", quiet=False)
    else:
        urllib.request.urlretrieve(GROUP_ZIP, "data/leaves.zip")
        downloaded = "data/leaves.zip"
    if not downloaded or not zipfile.is_zipfile("data/leaves.zip"):
        raise RuntimeError("The download is not a ZIP archive.")
    with zipfile.ZipFile("data/leaves.zip") as archive:
        if not all(any(name.startswith(f"leaves/{split}/") for name in archive.namelist()) for split in ["train", "val"]):
            raise ValueError("The ZIP must contain leaves/train and leaves/val.")
        for member in archive.infolist():
            if not (Path("data") / member.filename).resolve().is_relative_to(Path("data").resolve()):
                raise ValueError("Unsafe ZIP member.")
        archive.extractall("data")
    source_file.write_text(GROUP_ZIP)
print("Historical dataset supplied by the user is ready. This is not the current-group dataset.")''')

cell40 = "".join(nb["cells"][40]["source"])
if "assert train_ds.classes == val_ds.classes" not in cell40:
    cell40 = cell40.replace('val_ds = datasets.ImageFolder("data/leaves/val", transform=transform)',
                            'val_ds = datasets.ImageFolder("data/leaves/val", transform=transform)\nassert train_ds.classes == val_ds.classes')
cell40 = cell40.replace("DataLoader(train_ds, batch_size=32, shuffle=True)",
                        "DataLoader(train_ds, batch_size=32, shuffle=True,\n                          generator=torch.Generator().manual_seed(0))")
replace(40, cell40)

cell44 = "".join(nb["cells"][44]["source"])
cell44 = cell44.replace("loss_sum += loss.item()\n", "loss_sum += loss.item() * len(labels)\n")
cell44 = cell44.replace("loss_sum / len(train_loader)", "loss_sum / len(train_ds)")
replace(44, cell44)

replace(54, '''class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        folder = Path(folder)
        self.classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        self.samples = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for class_index, name in enumerate(self.classes):
            for path in sorted((folder / name).rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.samples.append((path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as original:
            image = original.convert("RGB")
        return self.transform(image), label


my_train_ds = LeafDataset("data/leaves/train", transform)
my_val_ds = LeafDataset("data/leaves/val", transform)
assert my_train_ds.classes == my_val_ds.classes
my_train_loader = DataLoader(my_train_ds, batch_size=32, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
my_val_loader = DataLoader(my_val_ds, batch_size=32, shuffle=False)
image, label = my_train_ds[-1]
print(my_train_ds.classes, "| train", len(my_train_ds), "| val", len(my_val_ds))
print("the last sample equals ImageFolder's:", torch.equal(image, train_ds[-1][0]) and label == train_ds[-1][1])''')

replace(56, '''class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


torch.manual_seed(0)
my_model = LeafNet().to(device)
print("parameters:", sum(p.numel() for p in my_model.parameters()))
print("output for 5 images:", tuple(my_model(torch.zeros(5, 3, 96, 96).to(device)).shape))''')

training = '''criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)
{history} = []
for epoch in range(15):
    my_model.train()
    loss_sum = 0.0
    for images, labels in {loader}:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(my_model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
    my_model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in my_val_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (my_model(images).argmax(dim=1) == labels).sum().item()
    mean_loss = loss_sum / len({loader}.dataset)
    accuracy = correct / len(my_val_ds)
    {history}.append({{"epoch": epoch + 1, "training_loss": mean_loss, "validation_accuracy": accuracy}})
    print(f"epoch {{epoch + 1:2d}}  training loss {{mean_loss:.4f}}  validation accuracy {{accuracy:.3f}}")'''
replace(58, training.format(history="task3_baseline_history", loader="my_train_loader"))

replace(60, '''aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aug_train_ds = LeafDataset("data/leaves/train", aug_transform)
aug_train_loader = DataLoader(aug_train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(0))


class LeafNetBN(LeafNet):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))''')
replace(61, 'torch.manual_seed(0)\nmy_model = LeafNetBN().to(device)\n' + training.format(history="task3_aug_bn_history", loader="aug_train_loader"))

export = "".join(nb["cells"][49]["source"])
export = export.replace('external_data=False)', 'external_data=False, dynamo=False, opset_version=17)')
replace(49, export)
note = '''## Execution notes and dataset provenance

The user supplied the historical course archive linked in Task 4, not the current
group's Google Drive ZIP. The data cell accepts that supplied source, and results
must be described as historical-data results. No current-group generalization is claimed.

Task 1 was executed in this notebook on CPU. Task 3's placeholder cells are filled
with the explicit Dataset, CNN and training loop. `leaf_pipeline.py` is the same
baseline with only the switches needed for the measured variants. `experiments.py`
runs the Task 2 demonstration, Task 3's four comparisons, then Task 4's ablations,
combinations and seed repeats. Its captured results are added below after execution.
Code cells without execution counts have not been run directly in this notebook.

REVIEW.md requires the student's own physical-phone check and screenshot. REPORT.md
requires the student's interpretation; generated tables and evidence support that writing.
'''
nb["cells"] = [c for c in nb["cells"] if not c.get("metadata", {}).get("execution_notes")]
nb["cells"].append({"cell_type": "markdown", "id": "execution-notes", "metadata": {"execution_notes": True},
                    "source": note.splitlines(keepends=True)})
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Completed Task 3 code and documented the historical dataset override.")

if __name__ == "__main__":
    # Execute structural checks, without retraining or fabricating cell outputs.
    import os
    os.chdir(ROOT)
    from run_notebook import execute_cells
    import torch
    torch.set_num_threads(4)
    namespace = execute_cells(nb, [1, 40, 54, 56, 60], {"__name__": "__notebook__", "Path": Path})
    from leaf_pipeline import LeafNet as ExperimentNet
    torch.manual_seed(0)
    reference = namespace["LeafNet"]()
    torch.manual_seed(0)
    experiment = ExperimentNet()
    assert list(reference.state_dict()) == list(experiment.state_dict())
    assert all(torch.equal(reference.state_dict()[k], experiment.state_dict()[k]) for k in reference.state_dict())
    x = torch.randn(2, 3, 96, 96)
    assert torch.equal(reference(x), experiment(x))
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Notebook and experiment baseline weights and predictions match exactly.")
