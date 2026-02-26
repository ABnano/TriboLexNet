#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_spotting_cnn.py

Keyword spotting for the word "RED" in piezoelectric acoustic recordings.

Standard-style pipeline:
------------------------
1) Sentence-level audio -> fixed-length segments (~0.6 s).
2) For each segment, compute a log-power spectrogram.
3) Train a small 2D CNN to classify segments as RED vs non-RED.
4) At test time, slide a window across each long sentence and obtain
   a probability curve p(RED | time); peaks are detections.

Novelty in this implementation:
--------------------------------
- We do NOT have manual word-level boundaries. Instead we:
  * build an initial RED template from simple sentences
    ("RED before story", "RED after the story"),
  * use normalized cross-correlation (NCC) to automatically locate the
    RED region in every "RED" sentence (weak supervision),
  * extract aligned RED segments across different sentences/tones and
    form a refined RED template.
- We then:
  * use the NCC-estimated centers to build a training dataset of RED
    and non-RED segments,
  * train a CNN on these segments,
  * demonstrate stability by overlaying CNN-detected RED segments
    from many sentences,
  * demonstrate dynamic keyword detection by plotting p(RED | t) for
    long sentences.
- Additional interpretability:
  * PCA of CNN feature embeddings for RED vs non-RED segments.
  * Gradient-based saliency maps on RED / non-RED spectrograms.
  * Sentence-level confusion matrix with percentage values.
  * Sentence-level ROC curve.

Outputs:
--------
All plots are saved under plots_red_cnn/:
- stage1_initial_red_template_waveform.png
- stage1_initial_red_template_spectrogram.png
- stage1_refined_red_template_waveform.png
- stage1_refined_red_template_spectrogram.png
- stage2_example_segments_spectrograms.png
- stage2_training_loss.png
- stage2_training_accuracy.png
- stage2_segment_embedding_pca.png
- stage2_saliency_red_example.png
- stage2_saliency_nonred_example.png
- stage3_detection_examples.png
- stage3_maxprob_hist_red_vs_notred.png
- stage3_overlay_detected_red_segments_waveform.png
- stage3_average_detected_red_spectrogram.png
- stage3_confusion_matrix_percent.png
- stage3_roc_curve.png
"""

import os
import argparse
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA

# Use non-interactive backend to avoid Tk errors on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# Basic IO & signal utilities
# ======================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current) arrays.

    Files use:
      - ';' as separator
      - ',' as decimal
      - 3 header lines, then data
      - columns: Time (s), Channel A (nA)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            df = pd.read_csv(
                f,
                sep=";",
                decimal=",",
                header=None,
                skiprows=3,
                names=["Time_s", "Current_nA"],
                engine="python",
            )
    t = df["Time_s"].to_numpy(dtype=np.float32)
    x = df["Current_nA"].to_numpy(dtype=np.float32)
    return t, x


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

    Returns:
        signals: list of 1D float32 arrays
        labels:  list of ints (1 for RED, 0 for NOT-RED)
        metas:   list of dicts with metadata
        sr:      sampling rate (Hz) estimated from first RED file
    """
    signals: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[Dict] = []

    def _load_one(zip_path: str, label: int) -> None:
        with zipfile.ZipFile(zip_path, "r") as z:
            members = sorted([m for m in z.namelist() if m.lower().endswith(".csv")])
            for m in members:
                parts = m.split("/")
                if len(parts) == 3:
                    sentence = parts[1]
                elif len(parts) == 2:
                    sentence = parts[0]
                else:
                    sentence = "UNKNOWN"
                t, x = read_signal_from_zip(zip_path, m)
                signals.append(x)
                labels.append(label)
                metas.append({
                    "zip": os.path.basename(zip_path),
                    "path": m,
                    "sentence": sentence,
                })

    _load_one(red_zip, label=1)
    _load_one(not_red_zip, label=0)

    # Estimate sampling rate from first RED recording
    with zipfile.ZipFile(red_zip, "r") as z:
        first_csv = sorted([m for m in z.namelist() if m.lower().endswith(".csv")])[0]
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
                            n_fft: int = 256,
                            hop_length: int = 128) -> np.ndarray:
    """
    Compute a log-power spectrogram using a Hann window.

    Returns:
        log_spec: np.ndarray of shape (n_freq, n_frames)
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < n_fft:
        pad = n_fft - n
        x = np.pad(x, (0, pad), mode="constant")
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


# ======================================================================
# NCC-based template and RED center estimation (weak supervision)
# ======================================================================

def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation between template and signal.
    Both inputs: 1D arrays.

    Returns:
        ncc: array of length len(signal) - len(template) + 1
             values roughly in [-1, 1].
    """
    t = np.asarray(template, dtype=np.float32)
    x = np.asarray(signal, dtype=np.float32)

    L = len(t)
    if L < 1 or len(x) < L:
        raise ValueError("Template must be shorter than signal.")

    # zero-mean template
    t = t - np.mean(t)
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-8:
        raise ValueError("Template norm too small.")

    # raw correlation
    c = np.correlate(x, t, mode="valid")  # length N-L+1

    # sliding stats of x
    x2 = x ** 2
    cumsum_x = np.concatenate(([0.0], np.cumsum(x)))
    cumsum_x2 = np.concatenate(([0.0], np.cumsum(x2)))
    N = len(x)

    sum_x = cumsum_x[L:] - cumsum_x[:-L]
    sum_x2 = cumsum_x2[L:] - cumsum_x2[:-L]

    var_seg = sum_x2 - (sum_x ** 2) / float(L)
    var_seg = np.maximum(var_seg, 1e-8)
    std_seg = np.sqrt(var_seg)

    ncc = c / (norm_t * std_seg)
    return ncc.astype(np.float32)


def extract_segment_around_center(signal: np.ndarray,
                                  center_idx: int,
                                  seg_length: int) -> np.ndarray:
    """
    Extract a segment of length seg_length around center_idx.
    Pads with zeros if needed.
    """
    x = np.asarray(signal, dtype=np.float32)
    half = seg_length // 2
    start = center_idx - half
    end = start + seg_length

    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0

    if end > len(x):
        pad_right = end - len(x)
        end = len(x)
    else:
        pad_right = 0

    seg = x[start:end]
    if pad_left > 0 or pad_right > 0:
        seg = np.pad(seg, (pad_left, pad_right), mode="constant")

    return seg


RED_TEMPLATE_SENTENCES = ("RED before story", "RED after the story")


def build_initial_template(signals: List[np.ndarray],
                           metas: List[Dict],
                           sr: float,
                           window_duration: float = 0.6) -> np.ndarray:
    """
    Build an initial RED template from the simple sentences:
        "RED before story" and "RED after the story".

    We locate the max-energy region in those sentences and
    average a fixed-duration window around it.
    """
    win_len = int(window_duration * sr)
    segs = []

    for x, m in zip(signals, metas):
        if m.get("sentence") not in RED_TEMPLATE_SENTENCES:
            continue
        x_norm = normalize_signal(x)
        abs_x = np.abs(x_norm)
        kernel_size = max(1, int(0.03 * sr))
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        env = np.convolve(abs_x, kernel, mode="same")
        centre = int(np.argmax(env))

        start = max(centre - win_len // 2, 0)
        end = start + win_len
        if end > len(x_norm):
            end = len(x_norm)
            start = max(0, end - win_len)
        seg = x_norm[start:end]
        if len(seg) == win_len:
            segs.append(seg)

    if not segs:
        raise RuntimeError("No segments found to build initial template.")

    template = np.mean(np.stack(segs, axis=0), axis=0)
    template = template - np.mean(template)
    norm = np.linalg.norm(template)
    if norm > 1e-8:
        template = template / norm
    return template.astype(np.float32)


def refine_template_and_get_centers(signals: List[np.ndarray],
                                    labels: np.ndarray,
                                    sr: float,
                                    initial_template: np.ndarray
                                    ) -> Tuple[np.ndarray, Dict[int, int], List[np.ndarray]]:
    """
    Use NCC with the initial template to:
      - find RED centers in all RED sentences,
      - extract aligned segments,
      - build a refined template.

    Returns:
        refined_template
        red_center_by_idx: dict mapping global index -> center sample idx
        red_segments: list of aligned RED segments used for refinement
    """
    seg_len = len(initial_template)
    red_indices = np.where(labels == 1)[0]
    segments = []
    red_center_by_idx: Dict[int, int] = {}

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        ncc = normalized_cross_correlation(initial_template, x)
        peak = int(np.argmax(ncc))
        center = peak + seg_len // 2
        red_center_by_idx[idx] = center
        seg = extract_segment_around_center(x, center, seg_len)
        segments.append(seg)

    if not segments:
        raise RuntimeError("No RED segments found in refinement step.")

    seg_arr = np.stack(segments, axis=0)
    refined = np.mean(seg_arr, axis=0)
    refined = refined - np.mean(refined)
    nrm = np.linalg.norm(refined)
    if nrm > 1e-8:
        refined = refined / nrm

    return refined.astype(np.float32), red_center_by_idx, segments


# ======================================================================
# Segment dataset & CNN model
# ======================================================================

class SegmentSpectrogramDataset(Dataset):
    """
    Dataset of fixed-length waveform segments with labels.
    Spectrograms are computed on-the-fly.
    """
    def __init__(self,
                 segments: List[np.ndarray],
                 labels: np.ndarray,
                 sr: float,
                 n_fft: int = 256,
                 hop_length: int = 128):
        self.segments = [normalize_signal(s) for s in segments]
        self.labels = labels.astype(np.float32)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        x = self.segments[idx]
        spec = compute_log_spectrogram(x, n_fft=self.n_fft, hop_length=self.hop_length)
        # Per-sample normalization
        mean = np.mean(spec)
        std = np.std(spec)
        spec = (spec - mean) / (std + 1e-6)
        X = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_freq, n_frames)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y


class KeywordCNN2D(nn.Module):
    """
    Compact 2D CNN for keyword spotting on log-spectrogram segments.
    """
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

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze(1)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return latent feature vector before the final classifier.
        Used for PCA-based interpretability.
        """
        return self._forward_features(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Build segment dataset from sentence-level signals
# ======================================================================

def create_segment_dataset(signals: List[np.ndarray],
                           labels: np.ndarray,
                           red_center_by_idx: Dict[int, int],
                           sr: float,
                           indices: np.ndarray,
                           seg_len: int,
                           pos_jitter_sec: float = 0.03,
                           num_pos_per_red: int = 2,
                           num_neg_per_sentence: int = 4
                           ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create a dataset of fixed-length segments.

    For RED sentences:
        - use NCC-estimated center as anchor,
        - generate 'num_pos_per_red' positive segments with small jitter,
        - generate negatives from regions far from the center.

    For non-RED sentences:
        - sample 'num_neg_per_sentence' random segments as negative.

    Returns:
        segments: list of 1D arrays
        seg_labels: array of 0/1
    """
    segments: List[np.ndarray] = []
    seg_labels: List[int] = []

    jitter_samples = int(pos_jitter_sec * sr)

    for idx in indices:
        x = normalize_signal(signals[idx])
        L = len(x)
        if labels[idx] == 1:
            if idx not in red_center_by_idx:
                center_base = L // 2
            else:
                center_base = red_center_by_idx[idx]

            # positive segments with jitter
            for _ in range(num_pos_per_red):
                delta = np.random.randint(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
                center = int(np.clip(center_base + delta, 0, L - 1))
                seg = extract_segment_around_center(x, center, seg_len)
                segments.append(seg)
                seg_labels.append(1)

            # negative segments from same RED sentence (away from RED)
            for _ in range(max(1, num_neg_per_sentence // 2)):
                for _try in range(20):
                    c = np.random.randint(seg_len // 2, L - seg_len // 2)
                    if abs(c - center_base) > seg_len:
                        seg = extract_segment_around_center(x, c, seg_len)
                        segments.append(seg)
                        seg_labels.append(0)
                        break

        else:
            # non-RED sentence: all are negative
            for _ in range(num_neg_per_sentence):
                c = np.random.randint(seg_len // 2, L - seg_len // 2)
                seg = extract_segment_around_center(x, c, seg_len)
                segments.append(seg)
                seg_labels.append(0)

    return segments, np.array(seg_labels, dtype=int)


# ======================================================================
# Training loop and plotting
# ======================================================================

def train_cnn(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: torch.device,
              num_epochs: int,
              lr: float,
              plots_dir: str) -> nn.Module:
    """
    Train the CNN and save training curves.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_state = None    # best model in terms of val_loss
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        # ------------------ train ------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ------------------ validate ------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss += loss.item() * xb.size(0)
                pred_labels = (preds >= 0.5).float()
                val_correct += (pred_labels == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot training curves (no grid lines)
    ensure_dir(plots_dir)
    epochs = np.arange(1, num_epochs + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / validation loss")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_training_loss.png"), dpi=300)
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(epochs, history["train_acc"], label="Train acc")
    ax2.plot(epochs, history["val_acc"], label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training / validation accuracy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_training_accuracy.png"), dpi=300)
    plt.close(fig)

    return model


# ======================================================================
# Plot helpers for analysis
# ======================================================================

def plot_template_waveform_and_spectrogram(template: np.ndarray,
                                           sr: float,
                                           plots_dir: str,
                                           prefix: str) -> None:
    ensure_dir(plots_dir)
    t = np.arange(len(template)) / sr

    # waveform
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, template)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"{prefix} waveform")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix.lower()}_waveform.png"), dpi=300)
    plt.close(fig)

    # spectrogram
    spec = compute_log_spectrogram(template, n_fft=256, hop_length=128)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{prefix} log-power spectrogram")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix.lower()}_spectrogram.png"), dpi=300)
    plt.close(fig)


def plot_example_segments_spectrograms(pos_segments: List[np.ndarray],
                                       neg_segments: List[np.ndarray],
                                       sr: float,
                                       plots_dir: str,
                                       max_examples: int = 4) -> None:
    """
    Show some RED vs non-RED segment spectrograms side by side.
    """
    ensure_dir(plots_dir)
    n_pos = min(max_examples, len(pos_segments))
    n_neg = min(max_examples, len(neg_segments))
    n_rows = max(n_pos, n_neg)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # positives
    for i in range(n_pos):
        spec = compute_log_spectrogram(pos_segments[i], n_fft=256, hop_length=128)
        ax = axes[i, 0]
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0],
        )
        ax.set_title("RED segment")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    # negatives
    for i in range(n_neg):
        spec = compute_log_spectrogram(neg_segments[i], n_fft=256, hop_length=128)
        ax = axes[i, 1]
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0],
        )
        ax.set_title("Non-RED segment")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_example_segments_spectrograms.png"), dpi=300)
    plt.close(fig)


def plot_segment_embedding_pca(dataset: Dataset,
                               model: KeywordCNN2D,
                               device: torch.device,
                               plots_dir: str) -> None:
    """
    Compute CNN feature embeddings for all segments in `dataset` and
    show a 2D PCA scatter for RED vs non-RED segments.
    """
    ensure_dir(plots_dir)
    if len(dataset) < 2:
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    feats_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            feats = model.forward_features(xb)
            feats_list.append(feats.cpu().numpy())
            labels_list.append(yb.cpu().numpy())

    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if feats.shape[0] < 2:
        return

    pca = PCA(n_components=2, random_state=0)
    emb = pca.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, name, color in [(0, "non-RED", "tab:orange"), (1, "RED", "tab:blue")]:
        idx = labels == float(cls)
        if np.any(idx):
            ax.scatter(emb[idx, 0], emb[idx, 1], s=20, alpha=0.7, label=name, c=color)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("CNN segment embedding (PCA of latent features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_segment_embedding_pca.png"), dpi=300)
    plt.close(fig)


def plot_saliency_for_segment(segment: np.ndarray,
                              label_name: str,
                              sr: float,
                              model: KeywordCNN2D,
                              device: torch.device,
                              plots_dir: str,
                              prefix: str) -> None:
    """
    Gradient-based saliency on the spectrogram of a single segment.
    Shows where the CNN is most sensitive.
    """
    ensure_dir(plots_dir)
    x = normalize_signal(segment)
    spec = compute_log_spectrogram(x, n_fft=256, hop_length=128)
    mean = np.mean(spec)
    std = np.std(spec)
    spec_norm = (spec - mean) / (std + 1e-6)

    X = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    X = X.to(device)
    X.requires_grad_(True)

    model.zero_grad()
    out = model(X)
    # For RED saliency, push output towards 1; for non-RED, towards 0
    if label_name.lower() == "red":
        target = out
    else:
        target = 1.0 - out
    target.backward()

    grad = X.grad.detach().cpu().numpy()[0, 0]
    sal = np.abs(grad)
    sal = sal / (sal.max() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # original log-spectrogram
    im0 = axes[0].imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0],
    )
    axes[0].set_title(f"{label_name} segment log-spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # saliency map
    im1 = axes[1].imshow(
        sal,
        origin="lower",
        aspect="auto",
        extent=[0, sal.shape[1] * 128 / sr, 0, sr / 2.0],
    )
    axes[1].set_title(f"Gradient saliency ({label_name})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"stage2_saliency_{prefix}.png"), dpi=300)
    plt.close(fig)


def sliding_window_probs(model: nn.Module,
                         signal: np.ndarray,
                         sr: float,
                         seg_len: int,
                         device: torch.device,
                         stride_sec: float = 0.05
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window across the signal and compute p(RED)
    for each window using the trained CNN.

    Returns:
        times: center times (s) for each window
        probs: predicted probabilities (0..1)
    """
    model.eval()
    x = normalize_signal(signal)
    L = len(x)
    stride = max(1, int(stride_sec * sr))

    times: List[float] = []
    probs: List[float] = []

    with torch.no_grad():
        for start in range(0, max(1, L - seg_len + 1), stride):
            end = start + seg_len
            if end > L:
                seg = np.pad(x[start:], (0, end - L), mode="constant")
            else:
                seg = x[start:end]

            spec = compute_log_spectrogram(seg, n_fft=256, hop_length=128)
            mean = np.mean(spec)
            std = np.std(spec)
            spec = (spec - mean) / (std + 1e-6)

            X = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            X = X.to(device)
            p = float(model(X).cpu().numpy()[0])

            center = start + seg_len / 2.0
            times.append(center / sr)
            probs.append(p)

    return np.array(times), np.array(probs)


def plot_detection_examples(signals: List[np.ndarray],
                            labels: np.ndarray,
                            metas: List[Dict],
                            indices: np.ndarray,
                            model: nn.Module,
                            sr: float,
                            seg_len: int,
                            device: torch.device,
                            plots_dir: str,
                            max_red: int = 2,
                            max_notred: int = 2) -> None:
    """
    Plot waveform + p(RED|t) for a few RED and non-RED test sentences.
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_idx = [i for i in indices if labels[i] == 1][:max_red]
    not_idx = [i for i in indices if labels[i] == 0][:max_notred]
    chosen = red_idx + not_idx
    if not chosen:
        return

    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    threshold = 0.5

    for row, idx in enumerate(chosen):
        x = normalize_signal(signals[idx])
        t = np.arange(len(x)) / sr
        sent = metas[idx].get("sentence", "UNKNOWN")
        lab = labels[idx]

        # waveform
        ax_w = axes[row, 0]
        ax_w.plot(t, x, linewidth=0.8)
        ax_w.set_title(f"Waveform ({'RED' if lab == 1 else 'non-RED'}): {sent}")
        ax_w.set_xlabel("Time (s)")
        ax_w.set_ylabel("Amplitude")

        # probs over time
        times, probs = sliding_window_probs(model, x, sr, seg_len, device, stride_sec=0.05)
        ax_p = axes[row, 1]
        ax_p.plot(times, probs, linewidth=1.0)
        ax_p.axhline(threshold, color="red", linestyle="--", label="Threshold")
        max_idx = int(np.argmax(probs))
        ax_p.axvline(times[max_idx], color="green", linestyle="--", label="Peak p(RED)")
        ax_p.set_title(f"p(RED | t) (max={probs[max_idx]:.3f})")
        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Probability")
        ax_p.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_detection_examples.png"), dpi=300)
    plt.close(fig)


def plot_maxprob_hist(signals: List[np.ndarray],
                      labels: np.ndarray,
                      indices: np.ndarray,
                      model: nn.Module,
                      sr: float,
                      seg_len: int,
                      device: torch.device,
                      plots_dir: str) -> None:
    """
    For each test sentence, compute max p(RED) over time and plot
    histograms for RED vs non-RED sentences.
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)

    max_red = []
    max_not = []

    for idx in indices:
        x = signals[idx]
        times, probs = sliding_window_probs(model, x, sr, seg_len, device)
        m = float(np.max(probs))
        if labels[idx] == 1:
            max_red.append(m)
        else:
            max_not.append(m)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(max_red, bins=10, alpha=0.7, density=True, label="RED sentences")
    ax.hist(max_not, bins=10, alpha=0.7, density=True, label="Non-RED sentences")
    ax.set_xlabel("Max p(RED) over sentence")
    ax.set_ylabel("Density")
    ax.set_title("Separability of RED vs non-RED by CNN max probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_maxprob_hist_red_vs_notred.png"), dpi=300)
    plt.close(fig)


def plot_overlay_detected_segments(signals: List[np.ndarray],
                                   labels: np.ndarray,
                                   indices: np.ndarray,
                                   model: nn.Module,
                                   sr: float,
                                   seg_len: int,
                                   device: torch.device,
                                   plots_dir: str) -> None:
    """
    For each RED test sentence, take the window with highest p(RED) and
    overlay these segments (waveform + average spectrogram).
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_indices = [i for i in indices if labels[i] == 1]
    if not red_indices:
        return

    segments = []

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        times, probs = sliding_window_probs(model, x, sr, seg_len, device)
        best = int(np.argmax(probs))
        center_time = times[best]
        center_idx = int(center_time * sr)
        seg = extract_segment_around_center(x, center_idx, seg_len)
        segments.append(seg)

    seg_arr = np.stack(segments, axis=0)
    mean_seg = np.mean(seg_arr, axis=0)
    std_seg = np.std(seg_arr, axis=0)
    tt = np.arange(seg_len) / sr

    # waveform overlay
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in seg_arr:
        ax.plot(tt, s, color="gray", alpha=0.3, linewidth=0.7)
    ax.plot(tt, mean_seg, color="blue", linewidth=2.0, label="Mean detected RED segment")
    ax.fill_between(
        tt,
        mean_seg - std_seg,
        mean_seg + std_seg,
        color="blue",
        alpha=0.2,
        label="Mean ± 1 std",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title("CNN-detected RED segments across test sentences")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_overlay_detected_red_segments_waveform.png"), dpi=300)
    plt.close(fig)

    # average spectrogram
    specs = []
    min_frames = None
    for s in segments:
        spec = compute_log_spectrogram(s, n_fft=256, hop_length=128)
        if min_frames is None or spec.shape[1] < min_frames:
            min_frames = spec.shape[1]
        specs.append(spec)
    specs_cropped = [sp[:, :min_frames] for sp in specs]
    spec_arr = np.stack(specs_cropped, axis=0)
    mean_spec = np.mean(spec_arr, axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        mean_spec,
        origin="lower",
        aspect="auto",
        extent=[0, min_frames * 128 / sr, 0, sr / 2.0],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Average log-power spectrogram of CNN-detected RED segments")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_average_detected_red_spectrogram.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix_percent(cm: np.ndarray,
                                  class_names: List[str],
                                  plots_dir: str,
                                  filename: str) -> None:
    """
    Plot confusion matrix normalised row-wise to percentages.
    """
    ensure_dir(plots_dir)
    cm = cm.astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100.0

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_perc, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Sentence-level confusion matrix (%)")

    for i in range(cm_perc.shape[0]):
        for j in range(cm_perc.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_perc[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_perc[i, j] > 50 else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close(fig)


def plot_roc_curve_sentence_level(y_true: np.ndarray,
                                  y_scores: np.ndarray,
                                  plots_dir: str,
                                  filename: str) -> None:
    """
    Plot ROC curve for sentence-level detection.
    """
    ensure_dir(plots_dir)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Sentence-level ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close(fig)


# ======================================================================
# Main pipeline
# ======================================================================

def run_pipeline(red_zip: str,
                 not_red_zip: str,
                 plots_dir: str,
                 num_epochs: int = 200) -> None:
    ensure_dir(plots_dir)

    print("Loading dataset...")
    signals, labels, metas, sr = load_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ≈ {sr:.1f} Hz).")

    # Normalize & crop to common length
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped all signals to length {sig_len} samples (~{sig_len/sr:.2f} s).")

    labels = np.array(labels, dtype=int)

    # Sentence-level train/test split
    idx_all = np.arange(len(signals))
    idx_train_sent, idx_test_sent = train_test_split(
        idx_all,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    print(f"Sentence-level split: train={len(idx_train_sent)}, test={len(idx_test_sent)}")

    # ------------------------------------------------------------------
    # Stage 1: NCC-based initial + refined template, RED centers
    # ------------------------------------------------------------------
    print("Stage 1: Building initial and refined RED templates...")
    initial_template = build_initial_template(signals, metas, sr, window_duration=0.6)
    plot_template_waveform_and_spectrogram(
        initial_template,
        sr,
        plots_dir,
        prefix="Stage1_Initial_RED_template",
    )

    refined_template, red_center_by_idx, red_segments = refine_template_and_get_centers(
        signals, labels, sr, initial_template
    )
    plot_template_waveform_and_spectrogram(
        refined_template,
        sr,
        plots_dir,
        prefix="Stage1_Refined_RED_template",
    )

    seg_len = len(refined_template)
    print(f"Template / segment length: {seg_len} samples (~{seg_len/sr:.2f} s).")

    # ------------------------------------------------------------------
    # Stage 2: Build segment dataset & train CNN
    # ------------------------------------------------------------------
    print("Stage 2: Building segment dataset for CNN training...")
    seg_train_signals, seg_train_labels = create_segment_dataset(
        signals=signals,
        labels=labels,
        red_center_by_idx=red_center_by_idx,
        sr=sr,
        indices=idx_train_sent,
        seg_len=seg_len,
        pos_jitter_sec=0.03,
        num_pos_per_red=2,
        num_neg_per_sentence=4,
    )
    print(
        f"Created {len(seg_train_signals)} segments for training (positives + negatives)."
    )
    print(
        f"Class balance: RED={seg_train_labels.sum()}, "
        f"non-RED={len(seg_train_labels) - seg_train_labels.sum()}",
    )

    # For visualization: some positives & negatives
    pos_segments = [s for s, y in zip(seg_train_signals, seg_train_labels) if y == 1]
    neg_segments = [s for s, y in zip(seg_train_signals, seg_train_labels) if y == 0]
    plot_example_segments_spectrograms(pos_segments, neg_segments, sr, plots_dir)

    # Segment-level train/validation split
    idx_seg_all = np.arange(len(seg_train_signals))
    idx_seg_train, idx_seg_val = train_test_split(
        idx_seg_all,
        test_size=0.2,
        stratify=seg_train_labels,
        random_state=0,
    )

    seg_train = [seg_train_signals[i] for i in idx_seg_train]
    seg_val = [seg_train_signals[i] for i in idx_seg_val]
    y_seg_train = seg_train_labels[idx_seg_train]
    y_seg_val = seg_train_labels[idx_seg_val]

    train_ds = SegmentSpectrogramDataset(seg_train, y_seg_train, sr)
    val_ds = SegmentSpectrogramDataset(seg_val, y_seg_val, sr)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    device = get_device()
    print("Using device:", device)

    model = KeywordCNN2D()
    model = train_cnn(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        lr=1e-3,
        plots_dir=plots_dir,
    )

    # Interpretability: PCA on latent features
    print("Stage 2b: Computing PCA of CNN latent features...")
    plot_segment_embedding_pca(train_ds, model, device, plots_dir)

    # Interpretability: saliency for one RED and one non-RED segment
    print("Stage 2c: Computing gradient-based saliency maps...")
    if len(pos_segments) > 0:
        plot_saliency_for_segment(
            pos_segments[0],
            label_name="RED",
            sr=sr,
            model=model,
            device=device,
            plots_dir=plots_dir,
            prefix="red_example",
        )
    if len(neg_segments) > 0:
        plot_saliency_for_segment(
            neg_segments[0],
            label_name="non-RED",
            sr=sr,
            model=model,
            device=device,
            plots_dir=plots_dir,
            prefix="nonred_example",
        )

    # ------------------------------------------------------------------
    # Stage 3: Dynamic detection on sentence-level test set
    # ------------------------------------------------------------------
    print("Stage 3: Evaluating dynamic RED detection on test sentences...")

    threshold = 0.5
    y_true_sent = labels[idx_test_sent]
    y_score_sent = []
    y_pred_sent = []

    for idx in idx_test_sent:
        x = signals[idx]
        times, probs = sliding_window_probs(model, x, sr, seg_len, device)
        max_prob = float(np.max(probs))
        y_score_sent.append(max_prob)
        y_pred_sent.append(1 if max_prob >= threshold else 0)

    y_score_sent = np.array(y_score_sent)
    y_pred_sent = np.array(y_pred_sent)

    print("\nSentence-level performance (CNN max probability > 0.5):")
    cm = confusion_matrix(y_true_sent, y_pred_sent)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))
    try:
        auc_val = roc_auc_score(y_true_sent, y_score_sent)
        print(f"ROC-AUC (sentence-level, using max p(RED)): {auc_val:.3f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    # Plots for dynamic behaviour, separability, and sentence-level metrics
    plot_detection_examples(
        signals,
        labels,
        metas,
        idx_test_sent,
        model,
        sr,
        seg_len,
        device,
        plots_dir,
    )
    plot_maxprob_hist(
        signals,
        labels,
        idx_test_sent,
        model,
        sr,
        seg_len,
        device,
        plots_dir,
    )
    plot_overlay_detected_segments(
        signals,
        labels,
        idx_test_sent,
        model,
        sr,
        seg_len,
        device,
        plots_dir,
    )
    plot_confusion_matrix_percent(
        cm,
        class_names=["non-RED", "RED"],
        plots_dir=plots_dir,
        filename="stage3_confusion_matrix_percent.png",
    )
    plot_roc_curve_sentence_level(
        y_true_sent,
        y_score_sent,
        plots_dir=plots_dir,
        filename="stage3_roc_curve.png",
    )

    print(f"All plots saved under: {os.path.abspath(plots_dir)}")
    print("Pipeline complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN-based keyword spotting for the word 'RED' in piezoelectric signals."
    )
    parser.add_argument(
        "--red_zip",
        type=str,
        default="Red_story_231125.zip",
        help="Path to zip file with RED sentences.",
    )
    parser.add_argument(
        "--not_red_zip",
        type=str,
        default="not_RED_speaking.zip",
        help="Path to zip file with non-RED sentences.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots_red_cnn",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs for the CNN.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        args.red_zip,
        args.not_red_zip,
        plots_dir=args.plots_dir,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
