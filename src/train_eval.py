from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class IndexedImageDataset(Dataset):
    def __init__(self, df, class_to_idx, image_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["image_path"]) as im:
            img = im.convert("RGB")
        return self.tf(img), self.class_to_idx[str(r["label"])], str(r["image_path"])


def _seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _save_checkpoint(model, path, classes, image_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": list(classes),
        "image_size": int(image_size),
        "architecture": "resnet18",
    }, str(path))


def _predict(model, loader, device):
    model.eval()
    ys, preds, probs_all, paths = [], [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            ys.extend(y.numpy().tolist())
            preds.extend(probs.argmax(1).cpu().numpy().tolist())
            probs_all.append(probs.cpu().numpy())
            paths.extend(list(p))
    return np.asarray(ys), np.asarray(preds), np.concatenate(probs_all, axis=0), paths


def expected_calibration_error(y_true, probs, bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def train_dataset(df, name, output_dir, model_dir, image_size=224, batch_size=32,
                  epochs=8, lr=3e-4, weight_decay=1e-4, num_workers=0, seed=42,
                  pretrained=True, device=None):
    _seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = df[df["valid_image"]].copy() if "valid_image" in df.columns else df.copy()
    df["label"] = df["label"].astype(str)
    labels = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(labels)}
    tr_df = df[df["final_split"] == "train"].copy()
    va_df = df[df["final_split"] == "val"].copy()
    te_df = df[df["final_split"] == "test"].copy()
    if min(len(tr_df), len(va_df), len(te_df)) == 0:
        raise RuntimeError("train/val/test must all be non-empty")

    print("Training %s" % name)
    print("  classes: %s" % labels)
    print("  train=%d val=%d test=%d device=%s" % (len(tr_df), len(va_df), len(te_df), device))

    tr = DataLoader(IndexedImageDataset(tr_df, class_to_idx, image_size, True), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(IndexedImageDataset(va_df, class_to_idx, image_size, False), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te = DataLoader(IndexedImageDataset(te_df, class_to_idx, image_size, False), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = _build_model(len(labels), pretrained=pretrained).to(device)
    counts = tr_df["label"].value_counts().reindex(labels).fillna(1).values.astype(np.float32)
    weights = torch.tensor((counts.sum() / counts) / (counts.sum() / counts).mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    model_dir = Path(model_dir); output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / (name + "_resnet18.pt")
    best_f1 = -1.0
    history = []

    for epoch in range(int(epochs)):
        model.train(); running = 0.0; nobs = 0
        for x, y, _ in tqdm(tr, desc="%s epoch %d/%d" % (name, epoch + 1, epochs)):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            running += float(loss.item()) * len(y); nobs += len(y)
        yv, pv, _, _ = _predict(model, va, device)
        vf1 = float(f1_score(yv, pv, average="macro", zero_division=0))
        scheduler.step(vf1)
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, nobs), "val_macro_f1": vf1, "lr": opt.param_groups[0]["lr"]})
        print("  epoch %d: loss=%.5f val_macro_f1=%.5f" % (epoch + 1, history[-1]["train_loss"], vf1))
        if vf1 > best_f1:
            best_f1 = vf1
            _save_checkpoint(model, best_path, labels, image_size)

    from .model import load_checkpoint
    best_model, _ = load_checkpoint(best_path, device=device)
    yt, yp, probs, paths = _predict(best_model, te, device)
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "log_loss": float(log_loss(yt, probs, labels=list(range(len(labels))))),
        "ece": expected_calibration_error(yt, probs),
        "classification_report": classification_report(yt, yp, target_names=labels, output_dict=True, zero_division=0),
    }
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / (name + "_confusion_matrix.csv"), encoding="utf-8-sig")
    pred_df = pd.DataFrame({
        "image_path": paths,
        "true_label": [labels[i] for i in yt],
        "predicted_label": [labels[i] for i in yp],
        "confidence": probs.max(axis=1),
    })
    pred_df.to_csv(output_dir / (name + "_test_predictions.csv"), index=False, encoding="utf-8-sig")
    with open(output_dir / (name + "_training.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "classes": labels, "history": history, "metrics": metrics, "checkpoint": str(best_path)}, f, ensure_ascii=False, indent=2)
    print("\n=== TEST RESULTS: %s ===" % name)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return best_path, metrics
