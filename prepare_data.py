"""Extract the user-supplied historical ZIP and freeze camera-like validation images."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip"
EXPECTED_SHA256 = "f7bbaa9aa161bd1c334e2fe3fe8b5e93af216ff1664f2571d3572eace555cf53"


def prepare(archive_path):
    archive_path = Path(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("This script reproduces the historical archive supplied by the user; ZIP hash differs. Use a separate project for other data.")
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (folder / member.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe ZIP member")
        archive.extractall(folder)
    leaves = folder / "leaves"
    for split in ["train", "val"]:
        if not (leaves / split).is_dir():
            raise ValueError("Expected leaves/train and leaves/val in the ZIP")
    (leaves / ".group_zip_url").write_text(SOURCE_URL, encoding="utf-8")
    # Seeded once, saved once: every variant sees the same transformed validation images.
    torch.manual_seed(123)
    camera_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.04),
    ])
    balance = {}
    for split in ["train", "val"]:
        balance[split] = {}
        for class_folder in sorted((leaves / split).iterdir()):
            if not class_folder.is_dir():
                continue
            images = sorted(p for p in class_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            balance[split][class_folder.name] = len(images)
            if split == "val":
                destination = folder / "camera_val" / class_folder.name
                destination.mkdir(parents=True, exist_ok=True)
                for i, image_path in enumerate(images):
                    with Image.open(image_path) as source:
                        image = camera_transform(source.convert("RGB"))
                    image.save(destination / f"{i:04d}.png")
    evidence = {"source": "User-supplied historical course archive, not the current-group Drive ZIP",
                "url": SOURCE_URL, "zip_sha256": digest,
                "class_counts": balance, "camera_validation_seed": 123,
                "camera_validation": "one saved crop/rotation/colour perturbation per validation image",
                "limitation": "Historical archive used as primary data at user's request; not an independent current-group comparison"}
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/data_provenance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    prepare(parser.parse_args().zip)
