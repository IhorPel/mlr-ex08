"""Small, readable Dataset, four-block CNN, and one explicit training loop."""
import copy
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def image_transform(geometric=False, colour=False):
    steps = []
    if geometric:
        steps += [transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
                  transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(20)]
    if colour:
        steps += [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Leaf dataset folder not found: {folder}")
        self.classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        self.samples = []
        for class_index, name in enumerate(self.classes):
            self.samples += [(p, class_index) for p in sorted((folder / name).rglob("*"))
                             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if len(self.classes) != 4 or not self.samples:
            raise ValueError("Expected four nonempty leaf classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as original:
            image = original.convert("RGB")
        return self.transform(image), label


class LeafNet(nn.Module):
    def __init__(self, batch_norm=False, dropout=0.0, global_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(32) if batch_norm else nn.Identity()
        self.bn3 = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.bn4 = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.global_pool = global_pool
        self.average = nn.AdaptiveAvgPool2d(1) if global_pool else nn.Identity()
        self.fc1 = nn.Linear(64 if global_pool else 64 * 6 * 6, 64)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.average(x).flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class LeafNetBN(LeafNet):
    def __init__(self):
        super().__init__(batch_norm=True)


def make_model(config):
    if config.get("transfer"):
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        if config["transfer"] == "frozen":
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.classifier[3].parameters():
                parameter.requires_grad = True
        return model
    return LeafNet(batch_norm=config.get("batch_norm", False),
                   dropout=config.get("dropout", 0.0), global_pool=config.get("global_pool", False))


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            prediction = logits.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            np.add.at(confusion, (labels.cpu().numpy(), prediction.cpu().numpy()), 1)
    n = len(loader.dataset)
    support = confusion.sum(axis=1)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(4, dtype=float), where=support > 0)
    return {"accuracy": correct / n, "balanced_accuracy": float(recall.mean()),
            "loss": loss_sum / n, "confusion_matrix": confusion.tolist(), "n": n}


def train_variant(config, seed, train_folder, val_folder, camera_folder, checkpoint, *, epochs=15, device="cpu"):
    seed_everything(seed)
    training = LeafDataset(train_folder, image_transform(config.get("geometric", False), config.get("colour", False)))
    validation = LeafDataset(val_folder, image_transform())
    camera = LeafDataset(camera_folder, image_transform())
    assert training.classes == validation.classes == camera.classes
    train_loader = DataLoader(training, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=32, shuffle=False, num_workers=0)
    camera_loader = DataLoader(camera, batch_size=32, shuffle=False, num_workers=0)
    model = make_model(config).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_class = torch.optim.AdamW if config.get("adamw") else torch.optim.Adam
    optimizer = optimizer_class(parameters, lr=config.get("learning_rate", 1e-3),
                                weight_decay=1e-3 if config.get("adamw") else 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if config.get("scheduler") else None
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    history = []
    best_accuracy, best_state, best_epoch = -1.0, None, None
    for epoch in range(epochs):
        model.train()
        if config.get("transfer") == "frozen":
            # Keep frozen batch-normalization statistics fixed, not just the weights.
            model.eval()
            model.classifier[3].train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        validation_result = evaluate(model, val_loader, device)
        if validation_result["accuracy"] > best_accuracy:
            best_accuracy = validation_result["accuracy"]
            best_epoch, best_state = epoch + 1, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch + 1, "training_loss": loss_sum / len(training),
                        "validation_accuracy": validation_result["accuracy"], "validation_loss": validation_result["loss"]})
        if scheduler:
            scheduler.step()
        print(f"{config['name']} seed={seed} epoch={epoch+1}/{epochs}: train loss {loss_sum/len(training):.4f}, "
              f"validation {validation_result['accuracy']:.4f}", flush=True)
    # Baseline matches Task 3's final epoch. best_epoch is a separate measured variant.
    selected_epoch = best_epoch if config.get("best_epoch") else epochs
    if config.get("best_epoch"):
        model.load_state_dict(best_state)
    clean = evaluate(model, val_loader, device)
    camera_result = evaluate(model, camera_loader, device)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "config": config, "classes": training.classes,
                "epoch": selected_epoch, "seed": seed}, checkpoint)
    return {"variant": config["name"], "seed": seed, "epochs": epochs, "selected_epoch": selected_epoch,
            "validation_accuracy": clean["accuracy"], "camera_accuracy": camera_result["accuracy"],
            "validation_balanced_accuracy": clean["balanced_accuracy"],
            "camera_balanced_accuracy": camera_result["balanced_accuracy"],
            "train_images": len(training), "validation_images": len(validation),
            "parameters": sum(p.numel() for p in model.parameters()),
            "clean_confusion": clean["confusion_matrix"], "camera_confusion": camera_result["confusion_matrix"]}, history
