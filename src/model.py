from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models


def _extract_checkpoint_parts(ckpt):
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        classes = ckpt.get("class_names") or ckpt.get("classes")
        image_size = int(ckpt.get("image_size", 224))
        if state is None and all(isinstance(k, str) for k in ckpt.keys()):
            # Raw state_dict fallback.
            if any(k.startswith("layer") or k.startswith("conv1") for k in ckpt.keys()):
                state = ckpt
        return state, classes, image_size
    return None, None, 224


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    ckpt = torch.load(str(path), map_location=device)
    state, classes, image_size = _extract_checkpoint_parts(ckpt)
    if state is None:
        raise ValueError("Unsupported checkpoint format: missing model_state_dict/state_dict")
    if not classes:
        # Infer class count from fc.weight but names are required for clinical prior mapping.
        n = int(state["fc.weight"].shape[0])
        classes = ["class_%d" % i for i in range(n)]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    meta = {"class_names": list(classes), "image_size": image_size}
    return model, meta
