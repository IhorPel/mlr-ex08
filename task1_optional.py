"""Measure the notebook's optional batch-size, split-seed and dropout exercises."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from run_notebook import execute_cells

torch.set_num_threads(4)
os.chdir(ROOT)
nb = json.loads((ROOT / "ex08_notebook.ipynb").read_text(encoding="utf-8"))
namespace = execute_cells(nb, [1, 4, 7, 16])
X, Y = namespace["X_t"], namespace["Y_t"]


def raw_training(batch_size, eta, maximum, target=None):
    W1 = torch.tensor(namespace["W1_start"], requires_grad=True)
    b1 = torch.tensor(namespace["b1_start"].ravel(), requires_grad=True)
    W2 = torch.tensor(namespace["W2_start"], requires_grad=True)
    b2 = torch.tensor(namespace["b2_start"].ravel(), requires_grad=True)
    rng = np.random.default_rng(0)
    for epoch in range(maximum):
        total = 0.0
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start+batch_size]
            prediction = torch.relu(X[batch] @ W1.T + b1) @ W2.T + b2
            loss = torch.mean(.5 * (prediction - Y[batch]) ** 2)
            total += loss.item() * len(batch)
            loss.backward()
            with torch.no_grad():
                for parameter in [W1, b1, W2, b2]:
                    parameter -= eta * parameter.grad
                    parameter.grad.zero_()
        mean = total / len(X)
        if target is not None and mean <= target:
            return {"epochs": epoch+1, "mean_half_mse": mean, "target_reached": True}
    return {"epochs": maximum, "mean_half_mse": mean, "target_reached": target is None}


results = {"batch1_eta001": raw_training(1, .01, 100),
           "fullbatch_eta01": raw_training(150, .1, 5000, target=.0569)}
dataset = TensorDataset(X.float(), Y.float())
rows = []
for split_seed, dropout in [(0, 0.), (1, 0.), (2, 0.), (0, .2)]:
    train, val = random_split(dataset, [120, 30], generator=torch.Generator().manual_seed(split_seed))
    torch.manual_seed(0)
    model = namespace["start_model"]()
    if dropout:
        # Same two Linear layers and initial values, inserting just one Dropout.
        model = nn.Sequential(model.layer1, nn.ReLU(), nn.Dropout(dropout), model.layer2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    train_loader = DataLoader(train, batch_size=10, shuffle=True, generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val, batch_size=30)
    best = (float("inf"), 0)
    for epoch in range(300):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xb, yb = next(iter(val_loader))
            value = nn.functional.mse_loss(model(xb), yb).item()
        if value < best[0]:
            best = (value, epoch+1)
    rows.append({"split_seed": split_seed, "dropout": dropout,
                 "epochs": 300, "best_validation_mse": best[0], "best_epoch": best[1]})
results["split_and_dropout"] = rows
(ROOT / "evidence/task1_optional.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
