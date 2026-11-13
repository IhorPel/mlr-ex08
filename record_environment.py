"""Record package versions without machine names, credentials or personal paths."""
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
names = ["torch", "torchvision", "numpy", "matplotlib", "pillow", "scikit-learn", "onnx", "onnxruntime", "onnxscript", "gdown", "qrcode", "pytest"]
environment = {"python": platform.python_version(), "platform": platform.system(),
               "packages": {name: importlib.metadata.version(name) for name in names},
               "experiment_threads": 4, "cuda_available": torch.cuda.is_available(),
               "training_device": "cpu", "initialization_and_shuffle_seeds": [0,1,2],
               "fixed_camera_validation_seed": 123,
               "browser_runtime": "onnxruntime-web 1.23.2, WASM, one thread"}
(root / "evidence/environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
print(json.dumps(environment, indent=2))
