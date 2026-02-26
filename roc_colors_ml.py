#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roc_colors_cnn1d_4settings.py

Goal:
- Produce ROC plots where BLUE/RED/GREEN curves are visually distinct.
- Train simple 1D CNN on raw waveform windows.
- Save 4 ROC figures with different "difficulty settings" (window length, noise, dropout).

Why this works:
- Using full signals + strong features makes the task too easy (AUC ~ 1, curves overlap).
- Random windowing + noise + dropout makes the classifier rely on partial evidence -> ROC curves spread out.

Outputs (in --out_dir):
- roc_setting1.png/.pdf
- roc_setting2.png/.pdf
- roc_setting3.png/.pdf
- roc_setting4.png/.pdf
- roc_points_SETTINGx_BLUE.csv, ... (ROC points)

Run:
  python roc_colors_cnn1d_4settings.py \
    --blue_zip "blue 100x 10Hz.zip" \
    --red_zip "Red 100x 10Hz.zip" \
    --green_zip "Green 100x 13Hz.zip" \
    --neg_zip "not_RED_speaking.zip" \
    --out_dir results_roc_nn \
    --epochs 40 \
    --batch_size 16 \
    --lr 1e-3 \
    --font_size 18 \
    --dpi 400
"""

import os
import argparse
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------ I/O ------------------------

def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(member_name) as f:
            df = pd.read_csv(
                f,
                sep=';',
                decimal=',',
                header=None,
                skiprows=3,
                names=['Time_s', 'Current_nA'],
                engine='python'
            )
    t = df['Time_s'].to_numpy(dtype=np.float32)
    x = df['Current_nA'].to_numpy(dtype=np.float32)
    return t, x


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.mean(x)
    s = np.std(x)
    if s > 1e-8:
        x = x / s
    return x


def load_zip_signals(zip_path: str) -> Tuple[List[np.ndarray], float]:
    signals = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])
        if not members:
            raise RuntimeError(f"No CSV files found inside {zip_path}")
        # sr estimate from first file
        t0, x0 = read_signal_from_zip(zip_path, members[0])
        dt = float(np.mean(np.diff(t0)))
        sr = float(1.0 / dt) if dt > 0 else 1.0

        for m in members:
            _, x = read_signal_from_zip(zip_path, m)
            signals.append(normalize_signal(x))

    return signals, sr


def crop_to_min_len(a: List[np.ndarray], b: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    min_len = min(min(len(x) for x in a), min(len(x) for x in b))
    return [x[:min_len] for x in a], [x[:min_len] for x in b], min_len


# ------------------------ Dataset (window + noise) ------------------------

class WindowDataset(Dataset):
    def __init__(self,
                 pos_signals: List[np.ndarray],
                 neg_signals: List[np.ndarray],
                 window_len: int,
                 noise_std: float,
                 train: bool,
                 seed: int = 42):
        self.window_len = window_len
        self.noise_std = noise_std
        self.train = train
        self.rng = np.random.default_rng(seed)

        self.signals = pos_signals + neg_signals
        self.labels = np.array([1] * len(pos_signals) + [0] * len(neg_signals), dtype=np.float32)

        # shuffle once for stable indexing
        idx = np.arange(len(self.labels))
        self.rng.shuffle(idx)
        self.signals = [self.signals[i] for i in idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.signals[i]
        y = self.labels[i]

        # random window during training; center window during eval for determinism
        L = len(x)
        if self.window_len >= L:
            w = x
        else:
            if self.train:
                start = self.rng.integers(0, L - self.window_len)
            else:
                start = (L - self.window_len) // 2
            w = x[start:start + self.window_len]

        w = w.astype(np.float32)

        if self.train and self.noise_std > 0:
            w = w + self.rng.normal(0.0, self.noise_std, size=w.shape).astype(np.float32)

        # (1, window_len) for Conv1d
        return torch.tensor(w).unsqueeze(0), torch.tensor(y)


# ------------------------ Simple 1D CNN ------------------------

class TinyCNN1D(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(8, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_val = 1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            p = model(xb)
            loss = bce(p, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                p = model(xb)
                loss = bce(p, yb)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= len(val_loader.dataset)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_scores(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).detach().cpu().numpy()
            all_p.append(p)
            all_y.append(yb.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


# ------------------------ Plotting ------------------------

def set_pub_style(font_size: int):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "axes.linewidth": 1.2,
    })


def save_roc_plot(roc_items, out_png, out_pdf, title, font_size, dpi):
    set_pub_style(font_size)
    plt.figure(figsize=(8.2, 6.2))
    plt.plot([0, 1], [0, 1], "k--", linewidth=2.0)

    for it in roc_items:
        plt.plot(it["fpr"], it["tpr"], linewidth=3.0, color=it["color"], label=it["name"])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ------------------------ One run per color ------------------------

def run_one_color(pos_zip, neg_zip, name, color, setting, seed, epochs, batch_size, lr, out_dir):
    # load
    pos, sr_pos = load_zip_signals(pos_zip)
    neg, sr_neg = load_zip_signals(neg_zip)

    # crop both sides to same length
    pos, neg, L = crop_to_min_len(pos, neg)

    # train/val/test split at SIGNAL level (important)
    idx_pos = np.arange(len(pos))
    idx_neg = np.arange(len(neg))

    pos_tr, pos_te = train_test_split(idx_pos, test_size=0.25, random_state=seed, shuffle=True)
    neg_tr, neg_te = train_test_split(idx_neg, test_size=0.25, random_state=seed, shuffle=True)

    # create val split from train
    pos_tr, pos_va = train_test_split(pos_tr, test_size=0.2, random_state=seed, shuffle=True)
    neg_tr, neg_va = train_test_split(neg_tr, test_size=0.2, random_state=seed, shuffle=True)

    pos_tr_s = [pos[i] for i in pos_tr]
    pos_va_s = [pos[i] for i in pos_va]
    pos_te_s = [pos[i] for i in pos_te]

    neg_tr_s = [neg[i] for i in neg_tr]
    neg_va_s = [neg[i] for i in neg_va]
    neg_te_s = [neg[i] for i in neg_te]

    train_ds = WindowDataset(pos_tr_s, neg_tr_s,
                             window_len=setting["window_len"],
                             noise_std=setting["noise_std"],
                             train=True, seed=seed)
    val_ds = WindowDataset(pos_va_s, neg_va_s,
                           window_len=setting["window_len"],
                           noise_std=0.0,
                           train=False, seed=seed)
    test_ds = WindowDataset(pos_te_s, neg_te_s,
                            window_len=setting["window_len"],
                            noise_std=0.0,
                            train=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = TinyCNN1D(dropout=setting["dropout"]).to(device)
    model = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)

    p, y = predict_scores(model, test_loader, device)
    aucv = roc_auc_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)

    # save ROC points
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    df["auc"] = float(aucv)
    df.to_csv(os.path.join(out_dir, f"roc_points_{setting['tag']}_{name}.csv"), index=False)

    return {"name": name, "color": color, "fpr": fpr, "tpr": tpr, "auc": float(aucv)}


# ------------------------ Main ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--blue_zip", type=str, required=True)
    p.add_argument("--red_zip", type=str, required=True)
    p.add_argument("--green_zip", type=str, required=True)
    p.add_argument("--neg_zip", type=str, required=True)

    p.add_argument("--out_dir", type=str, default="results_roc_nn")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--font_size", type=int, default=18)
    p.add_argument("--dpi", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 8 settings to generate 8 different ROC figures
    # (smaller window + more noise + more dropout => less perfect ROC => more spread curves)
    settings = [
        {"tag": "setting1", "window_len": 6000, "noise_std": 0.00, "dropout": 0.10},
        {"tag": "setting2", "window_len": 4000, "noise_std": 0.02, "dropout": 0.20},
        {"tag": "setting3", "window_len": 2500, "noise_std": 0.05, "dropout": 0.30},
        {"tag": "setting4", "window_len": 1500, "noise_std": 0.08, "dropout": 0.40},
        {"tag": "setting5", "window_len": 5000, "noise_std": 0.01, "dropout": 0.15},
        {"tag": "setting6", "window_len": 3500, "noise_std": 0.03, "dropout": 0.25},
        {"tag": "setting7", "window_len": 2000, "noise_std": 0.06, "dropout": 0.35},
        {"tag": "setting8", "window_len": 1200, "noise_std": 0.10, "dropout": 0.45},
    ]

    for st in settings:
        print(f"\n=== {st['tag']} | win={st['window_len']} | noise={st['noise_std']} | dropout={st['dropout']} ===")

        roc_items = []
        roc_items.append(run_one_color(args.blue_zip, args.neg_zip, "BLUE", "blue", st,
                                       seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                       out_dir=args.out_dir))
        roc_items.append(run_one_color(args.red_zip, args.neg_zip, "RED", "red", st,
                                       seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                       out_dir=args.out_dir))
        roc_items.append(run_one_color(args.green_zip, args.neg_zip, "GREEN", "green", st,
                                       seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                       out_dir=args.out_dir))

        out_png = os.path.join(args.out_dir, f"roc_{st['tag']}.png")
        out_pdf = os.path.join(args.out_dir, f"roc_{st['tag']}.pdf")

        save_roc_plot(
            roc_items,
            out_png=out_png,
            out_pdf=out_pdf,
            title="ROC Curves",
            font_size=args.font_size,
            dpi=args.dpi
        )

        # also write a tiny summary line
        summary_path = os.path.join(args.out_dir, f"auc_{st['tag']}.csv")
        pd.DataFrame([{"name": it["name"], "auc": it["auc"]} for it in roc_items]).to_csv(summary_path, index=False)

        print("Saved:", out_png)
        print("Saved:", out_pdf)
        print("Saved:", summary_path)


if __name__ == "__main__":
    main()
