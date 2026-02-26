#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_spotting_curriculum_v2.py

Curriculum-style keyword spotting for the word "RED" in piezoelectric
acoustic recordings.

New in this version:
--------------------
- Uses additional pure colour words as *hard negatives*:
  Blue, Green, Indigo, Orange, Violet, Yellow (same sensor, repeated
  at 10–13 Hz), to improve discrimination of RED vs other words.
- Trains in two curriculum stages and finally saves the trained CNN
  model as a .pth file for later testing.

Stage 1 (easy data):
--------------------
Inputs
  - Pure RED repetitions:
        * "5 times Red.zip"
        * "Red 100x 10Hz.zip"
        * "Red 100x 13Hz.zip"
  - Pure non-RED colour words (hard negatives), e.g.:
        * "Blue 100x 10Hz.zip"
        * "Green 100x 13Hz.zip"
        * "Indigo 100x 13Hz.zip"
        * "Orange 100x 13Hz.zip"
        * "Violet 100.zip"
        * "Yellow 100x 13Hz.zip"
  - Non-RED sentences:
        * "not_RED_speaking.zip"

Procedure
  - From pure RED recordings:
        envelope -> peaks -> fixed-length segments = positive examples.
  - From pure non-RED colour recordings:
        same envelope/peak pipeline, but labelled as negative examples.
  - From non-RED sentences:
        random segments as additional negatives.
  - Train a 2D CNN on log-power spectrograms of these segments.
  - Learn a clean RED template from pure RED segments.

Stage 2 (harder data, curriculum fine-tuning):
----------------------------------------------
Inputs
  - Sentences with RED:
        * "Red_story_231125.zip"
  - Sentences without RED:
        * "not_RED_speaking.zip"

Procedure
  - Use Stage-1 RED template + normalized cross-correlation (NCC)
    to locate RED centres in RED sentences (weak supervision).
  - Build a segment dataset from:
        * near-NCC-peak segments in RED sentences (positives),
        * segments far from the peak and from non-RED sentences (negatives).
  - Fine-tune the same CNN (curriculum).

Stage 3 (dynamic detection & stability analysis):
-------------------------------------------------
- For held-out test sentences, slide a window and compute p(RED | t).
- Visualizations:
    * Waveform + p(RED | t) plots (dynamic detection).
    * Histogram of max p(RED) for RED vs non-RED sentences
      (sentence-level separability).
    * Overlay of CNN-detected RED segments + average spectrogram
      (stability of the RED acoustic pattern).

The trained CNN model is saved as:
    red_keyword_cnn_curriculum.pth
(You can change the output path via --model_out.)
"""

import os
import argparse
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Non-interactive backend to avoid Tk errors on Windows
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


def estimate_sr_from_zip(zip_path: str) -> float:
    """
    Estimate sampling rate (Hz) from any CSV inside a zip.
    Assumes all recordings share the same sampling rate.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = [m for m in z.namelist() if m.lower().endswith('.csv')]
        if not members:
            raise RuntimeError(f"No CSV files found in {zip_path}")
        t, _ = read_signal_from_zip(zip_path, members[0])
    dt = np.mean(np.diff(t))
    return float(1.0 / dt)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale to unit variance."""
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x


def load_signals_from_zip(zip_path: str, category: str) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load all signals from a zip file.

    Returns:
        signals: list of 1D arrays
        metas:   list of dicts with metadata, including:
                 'zip', 'path', 'sentence', 'category'
    """
    signals: List[np.ndarray] = []
    metas: List[Dict] = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        members = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])
        for m in members:
            t, x = read_signal_from_zip(zip_path, m)
            parts = m.split('/')
            if len(parts) == 3:
                sentence = parts[1]
            elif len(parts) == 2:
                sentence = parts[0]
            else:
                sentence = "UNKNOWN"
            signals.append(x)
            metas.append({
                'zip': os.path.basename(zip_path),
                'path': m,
                'sentence': sentence,
                'category': category
            })

    return signals, metas


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


# ======================================================================
# Envelope, peak detection, NCC, segment extraction
# ======================================================================

def compute_envelope(x: np.ndarray, sr: float, smoothing_sec: float = 0.03) -> np.ndarray:
    """
    Compute a simple energy envelope using |x| and a moving average.
    """
    abs_x = np.abs(x)
    kernel_size = max(1, int(smoothing_sec * sr))
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    env = np.convolve(abs_x, kernel, mode='same')
    return env


def find_peaks_above_threshold(env: np.ndarray,
                               sr: float,
                               threshold_ratio: float = 0.4,
                               min_distance_sec: float = 0.25) -> List[int]:
    """
    Find local maxima in the envelope env that:
      - exceed threshold_ratio * max(env)
      - are separated by at least min_distance_sec seconds.
    """
    env = np.asarray(env, dtype=np.float32)
    N = len(env)
    if N < 3:
        return []

    threshold = threshold_ratio * float(np.max(env))
    min_dist = int(min_distance_sec * sr)

    candidates = []
    for i in range(1, N - 1):
        if env[i] >= env[i - 1] and env[i] >= env[i + 1] and env[i] > threshold:
            candidates.append(i)

    peaks: List[int] = []
    last_peak = -1e9
    for c in candidates:
        if c - last_peak >= min_dist:
            peaks.append(c)
            last_peak = c

    return peaks


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
        seg = np.pad(seg, (pad_left, pad_right), mode='constant')

    return seg


def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation between template and signal.
    Both inputs: 1D arrays.

    Returns:
        ncc: array of length len(signal) - len(template) + 1
             with values roughly in [-1, 1].
    """
    t = np.asarray(template, dtype=np.float32)
    x = np.asarray(signal, dtype=np.float32)

    L = len(t)
    if L < 1 or len(x) < L:
        raise ValueError("Template must be shorter than signal.")

    t = t - np.mean(t)
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-8:
        raise ValueError("Template norm too small.")

    # raw correlation
    c = np.correlate(x, t, mode='valid')  # length N-L+1

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


# ======================================================================
# CNN dataset and model
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
        mean = np.mean(spec)
        std = np.std(spec)
        spec = (spec - mean) / (std + 1e-6)
        X = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y


class KeywordCNN2D(nn.Module):
    """
    Compact 2D CNN for log-spectrogram keyword spotting.
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

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze(1)
        return x


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Building Stage-1 and Stage-2 segment datasets
# ======================================================================

def build_stage1_segments(pure_red_signals: List[np.ndarray],
                          notred_sentence_signals: List[np.ndarray],
                          pure_notred_word_signals: List[np.ndarray],
                          sr: float,
                          seg_len: int,
                          num_neg_per_notred_sentence: int = 4,
                          num_neg_per_notred_word: int = 3
                          ) -> Tuple[List[np.ndarray], np.ndarray,
                                     List[np.ndarray], List[np.ndarray]]:
    """
    Stage 1: build segments from pure RED, pure non-RED colour words,
    and non-RED sentences.

    - For each pure-RED signal:
        * envelope -> peaks -> segments (positives).
    - For each pure colour (non-RED) signal:
        * same envelope/peaks, but segments labelled as negatives.
    - For each non-RED sentence:
        * random segments as additional negatives.

    Returns:
        segments_all, labels_all, pos_segments, neg_segments
    """
    pos_segments: List[np.ndarray] = []
    neg_segments: List[np.ndarray] = []

    # positives from pure RED recordings
    for x in pure_red_signals:
        x_norm = normalize_signal(x)
        env = compute_envelope(x_norm, sr, smoothing_sec=0.03)
        peaks = find_peaks_above_threshold(env, sr,
                                           threshold_ratio=0.4,
                                           min_distance_sec=0.25)
        for p in peaks:
            seg = extract_segment_around_center(x_norm, p, seg_len)
            pos_segments.append(seg)

    # negatives from pure non-RED colour recordings (hard negatives)
    for x in pure_notred_word_signals:
        x_norm = normalize_signal(x)
        env = compute_envelope(x_norm, sr, smoothing_sec=0.03)
        peaks = find_peaks_above_threshold(env, sr,
                                           threshold_ratio=0.4,
                                           min_distance_sec=0.25)
        used = 0
        for p in peaks:
            seg = extract_segment_around_center(x_norm, p, seg_len)
            neg_segments.append(seg)
            used += 1
            if used >= num_neg_per_notred_word:
                break

    # negatives from non-RED sentences (varied background)
    for x in notred_sentence_signals:
        x_norm = normalize_signal(x)
        L = len(x_norm)
        if L < seg_len:
            continue
        for _ in range(num_neg_per_notred_sentence):
            center = np.random.randint(seg_len // 2, L - seg_len // 2)
            seg = extract_segment_around_center(x_norm, center, seg_len)
            neg_segments.append(seg)

    segments_all = pos_segments + neg_segments
    labels_all = np.array([1] * len(pos_segments) + [0] * len(neg_segments), dtype=int)

    return segments_all, labels_all, pos_segments, neg_segments


def refine_template_and_get_centers(signals: List[np.ndarray],
                                    labels: np.ndarray,
                                    sr: float,
                                    initial_template: np.ndarray
                                    ) -> Tuple[np.ndarray, Dict[int, int], List[np.ndarray]]:
    """
    Use NCC with an initial template to:
      - locate RED centers in all RED sentences,
      - extract aligned segments,
      - build a refined template.

    Returns:
        refined_template
        red_center_by_idx: dict mapping sentence index -> center sample idx
        red_segments: list of aligned RED segments used for refinement
    """
    seg_len = len(initial_template)
    red_indices = np.where(labels == 1)[0]
    segments: List[np.ndarray] = []
    red_center_by_idx: Dict[int, int] = {}

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        if len(x) < seg_len:
            continue
        ncc = normalized_cross_correlation(initial_template, x)
        peak = int(np.argmax(ncc))
        center = peak + seg_len // 2
        red_center_by_idx[idx] = center
        seg = extract_segment_around_center(x, center, seg_len)
        segments.append(seg)

    if not segments:
        raise RuntimeError("No RED segments found in sentences for refinement.")

    seg_arr = np.stack(segments, axis=0)
    refined = np.mean(seg_arr, axis=0)
    refined = refined - np.mean(refined)
    nrm = np.linalg.norm(refined)
    if nrm > 1e-8:
        refined = refined / nrm

    return refined.astype(np.float32), red_center_by_idx, segments


def create_stage2_segment_dataset(signals: List[np.ndarray],
                                  labels: np.ndarray,
                                  red_center_by_idx: Dict[int, int],
                                  sr: float,
                                  indices: np.ndarray,
                                  seg_len: int,
                                  pos_jitter_sec: float = 0.03,
                                  num_pos_per_red: int = 2,
                                  num_neg_per_sentence: int = 4
                                  ) -> Tuple[List[np.ndarray], np.ndarray,
                                             List[np.ndarray], List[np.ndarray]]:
    """
    Stage 2: create segment dataset from sentence-level signals.

    For RED sentences:
        - use NCC-estimated center as anchor,
        - generate 'num_pos_per_red' positive segments with small jitter,
        - generate negatives from regions far from the anchor.

    For non-RED sentences:
        - sample 'num_neg_per_sentence' random segments as negative.

    Returns:
        segments_all, labels_all, pos_segments, neg_segments
    """
    segments: List[np.ndarray] = []
    seg_labels: List[int] = []
    pos_segments: List[np.ndarray] = []
    neg_segments: List[np.ndarray] = []

    jitter_samples = int(pos_jitter_sec * sr)

    for idx in indices:
        x = normalize_signal(signals[idx])
        L = len(x)
        if L < seg_len:
            continue

        if labels[idx] == 1:
            if idx not in red_center_by_idx:
                center_base = L // 2
            else:
                center_base = red_center_by_idx[idx]

            # positive segments
            for _ in range(num_pos_per_red):
                delta = np.random.randint(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
                center = int(np.clip(center_base + delta, 0, L - 1))
                seg = extract_segment_around_center(x, center, seg_len)
                segments.append(seg)
                seg_labels.append(1)
                pos_segments.append(seg)

            # negatives from same RED sentence far from anchor
            for _ in range(max(1, num_neg_per_sentence // 2)):
                for _try in range(20):
                    c = np.random.randint(seg_len // 2, L - seg_len // 2)
                    if abs(c - center_base) > seg_len:
                        seg = extract_segment_around_center(x, c, seg_len)
                        segments.append(seg)
                        seg_labels.append(0)
                        neg_segments.append(seg)
                        break
        else:
            # non-RED sentence: all segments are negative
            for _ in range(num_neg_per_sentence):
                c = np.random.randint(seg_len // 2, L - seg_len // 2)
                seg = extract_segment_around_center(x, c, seg_len)
                segments.append(seg)
                seg_labels.append(0)
                neg_segments.append(seg)

    return segments, np.array(seg_labels, dtype=int), pos_segments, neg_segments


# ======================================================================
# Training & plotting helpers
# ======================================================================

def train_cnn(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: torch.device,
              num_epochs: int,
              lr: float,
              plots_dir: str,
              prefix: str) -> nn.Module:
    """
    Train the CNN and save training curves (loss+accuracy) with a prefix.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_state = None    # keep best model by val loss
    best_val_loss = float("inf")

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

        print(f"[{prefix} Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(plots_dir)
    epochs = np.arange(1, num_epochs + 1)

    # loss (no grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{prefix}: training / validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix}_training_loss.png"), dpi=300)
    plt.close(fig)

    # accuracy (no grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="Train")
    ax.plot(epochs, history["val_acc"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{prefix}: training / validation accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix}_training_accuracy.png"), dpi=300)
    plt.close(fig)

    return model


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
    fig.savefig(os.path.join(plots_dir, f"{prefix}_waveform.png"), dpi=300)
    plt.close(fig)

    # spectrogram
    spec = compute_log_spectrogram(template, n_fft=256, hop_length=128)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{prefix} log-power spectrogram")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix}_spectrogram.png"), dpi=300)
    plt.close(fig)


def plot_example_segments_spectrograms(pos_segments: List[np.ndarray],
                                       neg_segments: List[np.ndarray],
                                       sr: float,
                                       plots_dir: str,
                                       prefix: str,
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
            extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0]
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
            extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0]
        )
        ax.set_title("Non-RED segment")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(f"{prefix}: example segment spectrograms", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix}_example_segments_spectrograms.png"), dpi=300)
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
                seg = np.pad(x[start:], (0, end - L), mode='constant')
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
    Plot waveform + p(RED | t) for a few RED and non-RED test sentences.
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
        sent = metas[idx].get('sentence', 'UNKNOWN')
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
        _, probs = sliding_window_probs(model, x, sr, seg_len, device)
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

    segments: List[np.ndarray] = []

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
    ax.fill_between(tt, mean_seg - std_seg, mean_seg + std_seg,
                    color="blue", alpha=0.2, label="Mean ± 1 std")
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
        extent=[0, min_frames * 128 / sr, 0, sr / 2.0]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Average log-power spectrogram of CNN-detected RED segments")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_average_detected_red_spectrogram.png"), dpi=300)
    plt.close(fig)


# ======================================================================
# Main curriculum pipeline
# ======================================================================

def run_pipeline(pure_red_zips: List[str],
                 pure_nonred_word_zips: List[str],
                 red_zip: str,
                 not_red_zip: str,
                 plots_dir: str,
                 model_out: str,
                 seg_window_sec: float = 0.6,
                 epochs_stage1: int = 100,
                 epochs_stage2: int = 100) -> None:
    """
    Full curriculum pipeline, including training and saving the model.
    """
    ensure_dir(plots_dir)

    # Use RED sentence zip to estimate sampling rate
    sr = estimate_sr_from_zip(red_zip)
    print(f"Estimated sampling rate: {sr:.1f} Hz")

    # Load pure RED recordings
    pure_signals: List[np.ndarray] = []
    pure_metas: List[Dict] = []
    for path in pure_red_zips:
        if not os.path.exists(path):
            print(f"WARNING: pure RED zip not found and skipped: {path}")
            continue
        sigs, mets = load_signals_from_zip(path, category="pure_red")
        pure_signals.extend(sigs)
        pure_metas.extend(mets)
    print(f"Loaded {len(pure_signals)} pure-RED recordings from {len(pure_red_zips)} zip(s).")

    # Load pure non-RED colour recordings (hard negatives)
    pure_notred_signals: List[np.ndarray] = []
    pure_notred_metas: List[Dict] = []
    for path in pure_nonred_word_zips:
        if not os.path.exists(path):
            print(f"WARNING: pure non-RED colour zip not found and skipped: {path}")
            continue
        sigs, mets = load_signals_from_zip(path, category="pure_nonred_colour")
        pure_notred_signals.extend(sigs)
        pure_notred_metas.extend(mets)
    print(f"Loaded {len(pure_notred_signals)} pure non-RED colour recordings from {len(pure_nonred_word_zips)} zip(s).")

    # Load sentences with RED and without RED
    if not os.path.exists(red_zip):
        raise FileNotFoundError(f"RED sentence zip not found: {red_zip}")
    if not os.path.exists(not_red_zip):
        raise FileNotFoundError(f"non-RED sentence zip not found: {not_red_zip}")

    red_sent_signals, red_sent_metas = load_signals_from_zip(red_zip, category="sentence_red")
    notred_sent_signals, notred_sent_metas = load_signals_from_zip(not_red_zip, category="sentence_notred")
    print(f"Loaded {len(red_sent_signals)} RED sentences and {len(notred_sent_signals)} non-RED sentences.")

    # Combined sentence list
    signals_sent = red_sent_signals + notred_sent_signals
    metas_sent = red_sent_metas + notred_sent_metas
    labels_sent = np.array([1] * len(red_sent_signals) + [0] * len(notred_sent_signals), dtype=int)

    seg_len = int(seg_window_sec * sr)
    print(f"Segment/window length: {seg_len} samples (~{seg_len/sr:.2f} s).")

    # ------------------------------------------------------------------
    # Stage 1: curriculum on pure RED vs other colours + non-RED sentences
    # ------------------------------------------------------------------
    print("\n=== Stage 1: learning RED vs other colours (easy data) ===")
    segments1, labels1, pos1, neg1 = build_stage1_segments(
        pure_red_signals=pure_signals,
        notred_sentence_signals=notred_sent_signals,
        pure_notred_word_signals=pure_notred_signals,
        sr=sr,
        seg_len=seg_len,
        num_neg_per_notred_sentence=4,
        num_neg_per_notred_word=3
    )
    print(f"Stage1: segments={len(segments1)}, RED={labels1.sum()}, non-RED={len(labels1)-labels1.sum()}")

    # Stage-1 template from pure RED segments
    if not pos1:
        raise RuntimeError("No positive segments found in Stage 1.")
    template1 = np.mean(np.stack(pos1, axis=0), axis=0)
    template1 = template1 - np.mean(template1)
    nrm = np.linalg.norm(template1)
    if nrm > 1e-8:
        template1 = template1 / nrm
    plot_template_waveform_and_spectrogram(template1, sr, plots_dir,
                                           prefix="Stage1_pureRED_template")

    # Example spectrograms
    plot_example_segments_spectrograms(pos1, neg1, sr, plots_dir,
                                       prefix="Stage1", max_examples=4)

    # Train/val split for Stage1 segments
    idx_seg_all = np.arange(len(segments1))
    idx_seg_train, idx_seg_val = train_test_split(
        idx_seg_all,
        test_size=0.2,
        stratify=labels1,
        random_state=0
    )

    seg1_train = [segments1[i] for i in idx_seg_train]
    seg1_val = [segments1[i] for i in idx_seg_val]
    y1_train = labels1[idx_seg_train]
    y1_val = labels1[idx_seg_val]

    train_ds1 = SegmentSpectrogramDataset(seg1_train, y1_train, sr)
    val_ds1 = SegmentSpectrogramDataset(seg1_val, y1_val, sr)

    train_loader1 = DataLoader(train_ds1, batch_size=16, shuffle=True, num_workers=0)
    val_loader1 = DataLoader(val_ds1, batch_size=16, shuffle=False, num_workers=0)

    device = get_device()
    print("Using device:", device)

    model = KeywordCNN2D()
    model = train_cnn(model, train_loader1, val_loader1, device,
                      num_epochs=epochs_stage1, lr=1e-3,
                      plots_dir=plots_dir, prefix="Stage1")

    # ------------------------------------------------------------------
    # Stage 2: curriculum fine-tuning on sentences with/without RED
    # ------------------------------------------------------------------
    print("\n=== Stage 2: fine-tuning on full sentences ===")

    # Sentence-level split
    idx_all_sent = np.arange(len(signals_sent))
    idx_train_sent, idx_test_sent = train_test_split(
        idx_all_sent,
        test_size=0.2,
        stratify=labels_sent,
        random_state=42
    )
    print(f"Sentence-level split: train={len(idx_train_sent)}, test={len(idx_test_sent)}")

    # Refine template using sentences (NCC)
    refined_template, red_center_by_idx, red_segments_sentence = refine_template_and_get_centers(
        signals=signals_sent,
        labels=labels_sent,
        sr=sr,
        initial_template=template1
    )
    plot_template_waveform_and_spectrogram(refined_template, sr, plots_dir,
                                           prefix="Stage2_refined_template")

    # Build Stage-2 segment dataset from sentences
    segments2, labels2, pos2, neg2 = create_stage2_segment_dataset(
        signals=signals_sent,
        labels=labels_sent,
        red_center_by_idx=red_center_by_idx,
        sr=sr,
        indices=idx_train_sent,
        seg_len=seg_len,
        pos_jitter_sec=0.03,
        num_pos_per_red=2,
        num_neg_per_sentence=4
    )
    print(f"Stage2: segments={len(segments2)}, RED={labels2.sum()}, non-RED={len(labels2)-labels2.sum()}")

    plot_example_segments_spectrograms(pos2, neg2, sr, plots_dir,
                                       prefix="Stage2", max_examples=4)

    # Train/val split for Stage2 segments
    idx2_all = np.arange(len(segments2))
    idx2_train, idx2_val = train_test_split(
        idx2_all,
        test_size=0.2,
        stratify=labels2,
        random_state=1
    )
    seg2_train = [segments2[i] for i in idx2_train]
    seg2_val = [segments2[i] for i in idx2_val]
    y2_train = labels2[idx2_train]
    y2_val = labels2[idx2_val]

    train_ds2 = SegmentSpectrogramDataset(seg2_train, y2_train, sr)
    val_ds2 = SegmentSpectrogramDataset(seg2_val, y2_val, sr)

    train_loader2 = DataLoader(train_ds2, batch_size=16, shuffle=True, num_workers=0)
    val_loader2 = DataLoader(val_ds2, batch_size=16, shuffle=False, num_workers=0)

    # Fine-tune existing model (curriculum)
    model = train_cnn(model, train_loader2, val_loader2, device,
                      num_epochs=epochs_stage2, lr=5e-4,
                      plots_dir=plots_dir, prefix="Stage2")

    # ------------------------------------------------------------------
    # Save trained model for later testing
    # ------------------------------------------------------------------
    model_out_path = os.path.abspath(model_out)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "sr": sr,
            "seg_len": seg_len,
            "epochs_stage1": epochs_stage1,
            "epochs_stage2": epochs_stage2
        },
        model_out_path
    )
    print(f"\nSaved trained CNN model to: {model_out_path}")

    # ------------------------------------------------------------------
    # Stage 3: dynamic detection & stability on test sentences
    # ------------------------------------------------------------------
    print("\n=== Stage 3: dynamic RED detection on test sentences ===")

    threshold = 0.5
    y_true_sent = labels_sent[idx_test_sent]
    y_score_sent = []
    y_pred_sent = []

    for idx in idx_test_sent:
        x = signals_sent[idx]
        _, probs = sliding_window_probs(model, x, sr, seg_len, device)
        max_prob = float(np.max(probs))
        y_score_sent.append(max_prob)
        y_pred_sent.append(1 if max_prob >= threshold else 0)

    y_score_sent = np.array(y_score_sent)
    y_pred_sent = np.array(y_pred_sent)

    print("\nSentence-level performance (Stage2 CNN, max p(RED) > 0.5):")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_sent, y_pred_sent))
    print("\nClassification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))
    try:
        auc = roc_auc_score(y_true_sent, y_score_sent)
        print(f"ROC-AUC (sentence-level, using max p(RED)): {auc:.3f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    # Plots for dynamic behaviour & stability
    plot_detection_examples(signals_sent, labels_sent, metas_sent,
                            idx_test_sent, model, sr, seg_len, device, plots_dir)
    plot_maxprob_hist(signals_sent, labels_sent, idx_test_sent,
                      model, sr, seg_len, device, plots_dir)
    plot_overlay_detected_segments(signals_sent, labels_sent, idx_test_sent,
                                   model, sr, seg_len, device, plots_dir)

    print(f"\nAll plots saved under: {os.path.abspath(plots_dir)}")
    print("Curriculum pipeline complete.")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum CNN keyword spotting for the word 'RED'."
    )
    parser.add_argument(
        '--pure_red_zips',
        nargs='+',
        default=['5 times Red.zip', 'Red 100x 10Hz.zip', 'Red 100x 13Hz.zip'],
        help='Zip files containing pure RED repetitions.'
    )
    parser.add_argument(
        '--pure_nonred_word_zips',
        nargs='+',
        default=[
            'Blue 100x 10Hz.zip',
            'Green 100x 13Hz.zip',
            'Indigo 100x 13Hz.zip',
            'Orange 100x 13Hz.zip',
            'Violet 100.zip',
            'Yellow 100x 13Hz.zip'
        ],
        help='Zip files containing other colour words (non-RED).'
    )
    parser.add_argument(
        '--red_zip',
        type=str,
        default='Red_story_231125.zip',
        help='Zip file with sentences containing RED.'
    )
    parser.add_argument(
        '--not_red_zip',
        type=str,
        default='not_RED_speaking.zip',
        help='Zip file with sentences without the word RED.'
    )
    parser.add_argument(
        '--plots_dir',
        type=str,
        default='plots_red_curriculum',
        help='Directory to save all plots.'
    )
    parser.add_argument(
        '--model_out',
        type=str,
        default='red_keyword_cnn_curriculum.pth',
        help='Path to save the trained CNN model (.pth).'
    )
    parser.add_argument(
        '--epochs_stage1',
        type=int,
        default=100,
        help='Number of epochs for Stage-1 training (pure words).'
    )
    parser.add_argument(
        '--epochs_stage2',
        type=int,
        default=100,
        help='Number of epochs for Stage-2 fine-tuning (sentences).'
    )
    parser.add_argument(
        '--seg_window_sec',
        type=float,
        default=0.6,
        help='Window length in seconds for segments.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        pure_red_zips=args.pure_red_zips,
        pure_nonred_word_zips=args.pure_nonred_word_zips,
        red_zip=args.red_zip,
        not_red_zip=args.not_red_zip,
        plots_dir=args.plots_dir,
        model_out=args.model_out,
        seg_window_sec=args.seg_window_sec,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2
    )


if __name__ == "__main__":
    main()
