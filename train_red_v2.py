#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_detection.py

Detect the word "RED" in piezoelectric acoustic recordings.

Now also:
- Saves confusion matrix and ROC curve plots for each approach.
- Saves feature-importance bar plot for RandomForest.
- Saves training/validation loss curves for both CNNs.
"""

import os
import argparse
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# >>> IMPORTANT: use non-interactive backend so Tkinter is never touched
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    """Make a filesystem-safe name from a model name."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        .replace("-", "_")
    )


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str,
                          plots_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fname = os.path.join(plots_dir, f"cm_{safe_name(model_name)}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray,
                   y_score: np.ndarray,
                   model_name: str,
                   plots_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(plots_dir, f"roc_{safe_name(model_name)}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def plot_loss_curves(history: Dict[str, List[float]],
                     model_name: str,
                     plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"Training curves: {model_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(plots_dir, f"loss_{safe_name(model_name)}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current) arrays.
    """
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
    time = df['Time_s'].to_numpy(dtype=np.float32)
    current = df['Current_nA'].to_numpy(dtype=np.float32)
    return time, current


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale to unit variance."""
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x


def load_dataset(red_zip: str,
                 not_red_zip: str) -> Tuple[List[np.ndarray], List[int], List[Dict], float]:
    """
    Load all signals from the two zip files.
    """
    signals: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[Dict] = []

    def _load_one(zip_path: str, label: int) -> None:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])
            for m in members:
                parts = m.split('/')
                if len(parts) == 3:
                    sentence = parts[1]
                elif len(parts) == 2:
                    sentence = parts[0]
                else:
                    sentence = "UNKNOWN"

                time, current = read_signal_from_zip(zip_path, m)
                signals.append(current)
                labels.append(label)
                metas.append({
                    'zip': os.path.basename(zip_path),
                    'path': m,
                    'sentence': sentence
                })

    _load_one(red_zip, label=1)
    _load_one(not_red_zip, label=0)

    with zipfile.ZipFile(red_zip, 'r') as z:
        first_csv = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])[0]
        t, _ = read_signal_from_zip(red_zip, first_csv)
        dt = np.mean(np.diff(t))
        sr = float(1.0 / dt)

    return signals, labels, metas, sr


def crop_signals_to_min_length(signals: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """Crop all signals to the length of the shortest one."""
    min_len = min(len(s) for s in signals)
    cropped = [s[:min_len] for s in signals]
    return cropped, min_len


def compute_log_spectrogram(x: np.ndarray,
                            n_fft: int = 512,
                            hop_length: int = 256) -> np.ndarray:
    """
    Compute a simple log-power spectrogram using a Hann window.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < n_fft:
        pad = n_fft - n
        x = np.pad(x, (0, pad), mode='constant')
        n = len(x)
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, n - n_fft + 1, hop_length):
        frame = x[start:start + n_fft]
        frames.append(frame * window)
    frames = np.stack(frames, axis=0)  # (n_frames, n_fft)
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec) ** 2  # (n_frames, n_freq)
    power = power.T  # (n_freq, n_frames)
    log_spec = np.log10(power + 1e-12)
    return log_spec


# ---------------------------------------------------------------
# Approach 1 — Template / matched-filter baseline
# ---------------------------------------------------------------

RED_TEMPLATE_SENTENCES = ("RED before story", "RED after the story")


def build_red_template(signals: List[np.ndarray],
                       metas: List[Dict],
                       sr: float,
                       window_duration: float = 0.6) -> np.ndarray:
    """
    Build a template for the word "RED".
    """
    win_len = int(window_duration * sr)
    segments = []

    for s, m in zip(signals, metas):
        if m.get('sentence') not in RED_TEMPLATE_SENTENCES:
            continue
        x = normalize_signal(s)
        abs_x = np.abs(x)
        kernel_size = max(1, int(0.03 * sr))  # ~30 ms
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        env = np.convolve(abs_x, kernel, mode='same')
        centre = int(np.argmax(env))

        start = max(centre - win_len // 2, 0)
        end = start + win_len
        if end > len(x):
            end = len(x)
            start = max(0, end - win_len)
        seg = x[start:end]
        if len(seg) == win_len:
            segments.append(seg)

    if not segments:
        raise RuntimeError("No segments found for building RED template. "
                           "Check RED_TEMPLATE_SENTENCES vs folder names.")

    template = np.mean(np.stack(segments, axis=0), axis=0)
    template = template - np.mean(template)
    norm = np.linalg.norm(template)
    if norm > 1e-8:
        template = template / norm
    return template.astype(np.float32)


def max_template_correlation(template: np.ndarray, signal: np.ndarray) -> float:
    """
    Compute maximum correlation between template and signal.
    """
    x = normalize_signal(signal)
    corr = np.correlate(x, template, mode='valid')
    return float(np.max(corr))


# ---------------------------------------------------------------
# Approach 2 — Hand-crafted features + RandomForest
# ---------------------------------------------------------------

def extract_handcrafted_features(signal: np.ndarray,
                                 sr: float,
                                 n_bands: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    Extract time-domain and spectral features.
    """
    x = normalize_signal(signal)
    feats: List[float] = []
    names: List[str] = []

    # 1) Time-domain
    feats.append(float(np.mean(x)));      names.append('time_mean')
    feats.append(float(np.std(x)));       names.append('time_std')
    feats.append(float(np.max(x)));       names.append('time_max')
    feats.append(float(np.min(x)));       names.append('time_min')
    feats.append(float(np.ptp(x)));       names.append('time_ptp')
    feats.append(float(np.sqrt(np.mean(x ** 2)))); names.append('time_rms')

    # 2) Absolute signal
    abs_x = np.abs(x)
    feats.append(float(np.mean(abs_x)));  names.append('abs_mean')
    feats.append(float(np.std(abs_x)));   names.append('abs_std')
    feats.append(float(np.max(abs_x)));   names.append('abs_max')

    # 3) First difference
    dx = np.diff(x)
    feats.append(float(np.mean(dx)));     names.append('diff_mean')
    feats.append(float(np.std(dx)));      names.append('diff_std')
    feats.append(float(np.max(dx)));      names.append('diff_max')
    feats.append(float(np.min(dx)));      names.append('diff_min')

    # 4) Frequency-domain
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mag = np.abs(np.fft.rfft(x))
    total = np.sum(mag) + 1e-12
    mag_norm = mag / total

    max_f = sr / 2.0
    band_edges = np.linspace(0.0, max_f, n_bands + 1)
    for i in range(n_bands):
        f0, f1 = band_edges[i], band_edges[i + 1]
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        if len(idx) == 0:
            e = 0.0
        else:
            e = float(np.sum(mag_norm[idx]))
        feats.append(e)
        names.append(f'band_{i:02d}_{int(f0)}-{int(f1)}Hz')

    return np.array(feats, dtype=np.float32), names


def build_feature_matrix(signals: List[np.ndarray],
                         sr: float,
                         n_bands: int = 32) -> Tuple[np.ndarray, List[str]]:
    feat_list = []
    names_ref: List[str] = []
    for i, s in enumerate(signals):
        feats, names = extract_handcrafted_features(s, sr, n_bands=n_bands)
        feat_list.append(feats)
        if i == 0:
            names_ref = names
    X = np.stack(feat_list, axis=0)
    return X, names_ref


# ---------------------------------------------------------------
# Approach 3 — 2D CNN on log-power spectrogram
# ---------------------------------------------------------------

class SpectrogramDataset(Dataset):
    def __init__(self,
                 signals: List[np.ndarray],
                 labels: np.ndarray,
                 n_fft: int = 512,
                 hop_length: int = 256):
        self.signals = signals
        self.labels = labels.astype(np.float32)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        x = self.signals[idx].astype(np.float32)
        spec = compute_log_spectrogram(x, n_fft=self.n_fft, hop_length=self.hop_length)
        mean = np.mean(spec)
        std = np.std(spec)
        spec = (spec - mean) / (std + 1e-6)
        X_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X_tensor, y_tensor


class SmallCNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze(1)
        return x


# ---------------------------------------------------------------
# Approach 4 — 1D CNN on waveform
# ---------------------------------------------------------------

class WaveformDataset(Dataset):
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray):
        self.signals = [normalize_signal(s) for s in signals]
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        x = self.signals[idx].astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, L)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_tensor, y_tensor


class SmallCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze(1)
        return x


# ---------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------

def report_metrics(name: str,
                   y_true: np.ndarray,
                   y_score: np.ndarray,
                   plots_dir: str = None) -> None:
    """
    Print performance and optionally save confusion matrix + ROC plots.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        auc = None

    if plots_dir is not None:
        ensure_dir(plots_dir)
        plot_confusion_matrix(y_true, y_pred, name, plots_dir)
        if auc is not None:
            plot_roc_curve(y_true, y_score, name, plots_dir)


def train_torch_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      device: torch.device,
                      num_epochs: int = 20,
                      lr: float = 1e-3) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Generic training loop for BCE loss."""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------

def run_experiments(red_zip: str,
                    not_red_zip: str,
                    plots_dir: str = "plots",
                    run_template: bool = True,
                    run_rf: bool = True,
                    run_cnn2d: bool = True,
                    run_cnn1d: bool = True) -> None:
    ensure_dir(plots_dir)

    print("Loading dataset...")
    signals, labels, metas, sr = load_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ~ {sr:.1f} Hz).")

    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped all signals to common length: {sig_len} samples (~{sig_len/sr:.2f} s).")

    labels = np.array(labels, dtype=int)

    idx_all = np.arange(len(signals))
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    X_train = [signals[i] for i in idx_train]
    y_train = labels[idx_train]
    X_test = [signals[i] for i in idx_test]
    y_test = labels[idx_test]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 1) Template + LogisticRegression
    if run_template:
        print("\n=== Running Approach 1: Template + LogisticRegression ===")
        template = build_red_template(signals, metas, sr)
        train_feat = np.array([max_template_correlation(template, s) for s in X_train],
                              dtype=np.float32).reshape(-1, 1)
        test_feat = np.array([max_template_correlation(template, s) for s in X_test],
                             dtype=np.float32).reshape(-1, 1)

        clf_lr = LogisticRegression()
        clf_lr.fit(train_feat, y_train)
        y_score = clf_lr.predict_proba(test_feat)[:, 1]
        report_metrics("Template + LogisticRegression", y_test, y_score, plots_dir=plots_dir)

    # 2) RF + handcrafted features
    if run_rf:
        print("\n=== Running Approach 2: Hand-crafted features + RandomForest ===")
        X_all_feats, feat_names = build_feature_matrix(signals, sr, n_bands=32)
        X_train_feats = X_all_feats[idx_train]
        X_test_feats = X_all_feats[idx_test]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feats)
        X_test_scaled = scaler.transform(X_test_feats)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        y_score_rf = rf.predict_proba(X_test_scaled)[:, 1]
        report_metrics("RandomForest + handcrafted features", y_test, y_score_rf, plots_dir=plots_dir)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop 20 most important features:")
        top_k = min(20, len(indices))
        for rank in range(top_k):
            i = indices[rank]
            print(f"{rank+1:2d}. {feat_names[i]:30s}  importance={importances[i]:.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        top_idx = indices[:top_k]
        ax.bar(range(top_k), importances[top_idx])
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([feat_names[i] for i in top_idx], rotation=90)
        ax.set_ylabel("Importance")
        ax.set_title("RandomForest Feature Importance (top 20)")
        fig.tight_layout()
        fname = os.path.join(plots_dir, f"feat_importance_{safe_name('RandomForest + handcrafted features')}.png")
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    # 3) 2D CNN on spectrogram
    if run_cnn2d:
        print("\n=== Running Approach 3: 2D CNN on log-power spectrogram ===")
        idx_train_sub, idx_val = train_test_split(
            np.arange(len(X_train)),
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        X_train_sub = [X_train[i] for i in idx_train_sub]
        y_train_sub = y_train[idx_train_sub]
        X_val = [X_train[i] for i in idx_val]
        y_val = y_train[idx_val]

        train_ds = SpectrogramDataset(X_train_sub, y_train_sub)
        val_ds = SpectrogramDataset(X_val, y_val)
        test_ds = SpectrogramDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

        device = get_device()
        print("Using device:", device)

        model2d = SmallCNN2D()
        model2d, hist2d = train_torch_model(model2d, train_loader, val_loader, device, num_epochs=20, lr=1e-3)
        plot_loss_curves(hist2d, "2D CNN on log-power spectrogram", plots_dir)

        model2d.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model2d(xb)
                all_scores.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        y_score_2d = np.concatenate(all_scores, axis=0)
        y_true_2d = np.concatenate(all_labels, axis=0)
        report_metrics("2D CNN on log-power spectrogram", y_true_2d, y_score_2d, plots_dir=plots_dir)

    # 4) 1D CNN on waveform
    if run_cnn1d:
        print("\n=== Running Approach 4: 1D CNN on waveform ===")
        idx_train_sub, idx_val = train_test_split(
            np.arange(len(X_train)),
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        X_train_sub = [X_train[i] for i in idx_train_sub]
        y_train_sub = y_train[idx_train_sub]
        X_val = [X_train[i] for i in idx_val]
        y_val = y_train[idx_val]

        train_ds = WaveformDataset(X_train_sub, y_train_sub)
        val_ds = WaveformDataset(X_val, y_val)
        test_ds = WaveformDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

        device = get_device()
        print("Using device:", device)

        model1d = SmallCNN1D()
        model1d, hist1d = train_torch_model(model1d, train_loader, val_loader, device, num_epochs=20, lr=1e-3)
        plot_loss_curves(hist1d, "1D CNN on waveform", plots_dir)

        model1d.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model1d(xb)
                all_scores.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        y_score_1d = np.concatenate(all_scores, axis=0)
        y_true_1d = np.concatenate(all_labels, axis=0)
        report_metrics("1D CNN on waveform", y_true_1d, y_score_1d, plots_dir=plots_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect the word 'RED' in piezoelectric acoustic signals.")
    parser.add_argument('--red_zip', type=str, default='Red_story_231125.zip',
                        help='Path to zip with RED stories.')
    parser.add_argument('--not_red_zip', type=str, default='not_RED_speaking.zip',
                        help='Path to zip with non-RED stories.')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots.')
    parser.add_argument('--no_template', action='store_true', help='Disable template baseline.')
    parser.add_argument('--no_rf', action='store_true', help='Disable RandomForest baseline.')
    parser.add_argument('--no_cnn2d', action='store_true', help='Disable 2D CNN.')
    parser.add_argument('--no_cnn1d', action='store_true', help='Disable 1D CNN.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(
        red_zip=args.red_zip,
        not_red_zip=args.not_red_zip,
        plots_dir=args.plots_dir,
        run_template=not args.no_template,
        run_rf=not args.no_rf,
        run_cnn2d=not args.no_cnn2d,
        run_cnn1d=not args.no_cnn1d
    )


if __name__ == '__main__':
    main()

