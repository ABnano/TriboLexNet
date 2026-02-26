#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_spotting_ml.py

Keyword spotting for the word "RED" in piezoelectric acoustic recordings,
using a *classical machine-learning* pipeline:

1) Sentence-level signals (≈10 s) from two zip files:
       - RED sentences  (contain the word "red")
       - non-RED sentences (no "red")
2) Build an initial RED template from simple sentences:
       "RED before story", "RED after the story".
3) Refine the template and estimate a RED "centre" in each RED sentence
   by normalized cross-correlation (weak supervision).
4) Around those centres, extract fixed-length segments (~0.30 s).
   From RED and non-RED sentences we also extract negative segments.
5) For each segment we compute hand-crafted features:
       - max normalized correlation with template
       - RMS amplitude, energy, zero-crossing rate
       - spectral centroid
       - average log-power in 4 frequency bands (0–400, 400–800,
         800–1600, 1600–4000 Hz)
6) Train a Random Forest classifier on these segment features.
   Tune the decision threshold on a validation split using F1-score.
7) At test time, slide a 0.30 s window over each sentence, compute
   features, and get p(RED | window).
   The maximum p(RED) over time is used for sentence-level prediction
   and the argmax gives the detected RED time.

Outputs (all under plots_red_ml/ by default):
---------------------------------------------
- stage1_initial_template_waveform.png
- stage1_initial_template_spectrogram.png
- stage1_refined_template_waveform.png
- stage1_refined_template_spectrogram.png
- stage2_example_segments_spectrograms.png
- stage2_feature_importances.png
- stage2_segment_confusion_matrix_percent.png
- stage3_detection_examples.png
- stage3_maxprob_hist_red_vs_notred.png
- stage3_overlay_detected_red_segments_waveform.png
- stage3_sentence_confusion_matrix_percent.png
"""

import os
import argparse
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
)

# Avoid Tk errors on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current) arrays.

    Files use:
      - ';' as separator
      - ',' as decimal separator
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


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale to unit variance."""
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x


def load_sentence_dataset(red_zip: str,
                          not_red_zip: str) -> Tuple[List[np.ndarray], np.ndarray,
                                                     List[Dict], float]:
    """
    Load all sentence-level signals from the two zip files.

    Returns:
        signals: list of 1D float32 arrays
        labels:  np.ndarray of ints (1 for RED, 0 for non-RED)
        metas:   list of dicts with metadata
        sr:      sampling rate (Hz), estimated from first RED file
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
                t, x = read_signal_from_zip(zip_path, m)
                signals.append(x)
                labels.append(label)
                metas.append({
                    'zip': os.path.basename(zip_path),
                    'path': m,
                    'sentence': sentence
                })

    _load_one(red_zip, label=1)
    _load_one(not_red_zip, label=0)

    # Estimate sampling rate from first RED recording
    with zipfile.ZipFile(red_zip, 'r') as z:
        first_csv = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])[0]
        t, _ = read_signal_from_zip(red_zip, first_csv)
        dt = float(np.mean(np.diff(t)))
        sr = float(1.0 / dt)

    return signals, np.array(labels, dtype=int), metas, sr


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
        x = np.pad(x, (0, n_fft - n), mode='constant')
        n = len(x)

    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, n - n_fft + 1, hop_length):
        frame = x[start:start + n_fft]
        frames.append(frame * window)
    frames = np.stack(frames, axis=0)  # (n_frames, n_fft)
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec) ** 2          # (n_frames, n_freq)
    power = power.T                    # (n_freq, n_frames)
    log_spec = np.log10(power + 1e-12)
    return log_spec


# ============================================================
# NCC-based template and RED centres (weak supervision)
# ============================================================

def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation between template and signal (1D).

    Returns:
        ncc: array of length len(signal) - len(template) + 1
             values roughly in [-1, 1]
    """
    t = np.asarray(template, dtype=np.float32)
    x = np.asarray(signal, dtype=np.float32)

    L = len(t)
    if L < 1 or len(x) < L:
        raise ValueError("Template must be shorter than or equal to signal.")

    # zero-mean template
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
    out_len = N - L + 1

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
        seg = np.pad(seg, (pad_left, pad_right), mode='constant')

    return seg


RED_TEMPLATE_SENTENCES = ("RED before story", "RED after the story")


def build_initial_template(signals: List[np.ndarray],
                           metas: List[Dict],
                           sr: float,
                           window_duration: float = 0.30) -> np.ndarray:
    """
    Build an initial RED template from simple sentences:
        "RED before story" and "RED after the story".

    We locate the max-energy region in those sentences and
    average a fixed-duration window around it.

    window_duration was chosen ~0.30 s based on analysis of
    'Red 100x 10Hz' single-word recordings (typical RED burst
    ~0.10–0.15 s), adding some context.
    """
    win_len = int(window_duration * sr)
    segs = []

    for x, m in zip(signals, metas):
        if m.get('sentence') not in RED_TEMPLATE_SENTENCES:
            continue
        x_norm = normalize_signal(x)
        abs_x = np.abs(x_norm)
        kernel_size = max(1, int(0.01 * sr))  # 10 ms smoothing
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        env = np.convolve(abs_x, kernel, mode='same')
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
      - find RED centres in all RED sentences,
      - extract aligned segments,
      - build a refined template.

    Returns:
        refined_template
        red_center_by_idx: dict mapping sentence index -> centre sample idx
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
        centre = peak + seg_len // 2
        red_center_by_idx[idx] = centre
        seg = extract_segment_around_center(x, centre, seg_len)
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


# ============================================================
# Feature extraction for classical ML
# ============================================================

FEATURE_NAMES = [
    "max_corr_with_template",
    "segment_rms",
    "segment_energy",
    "zero_crossing_rate",
    "spectral_centroid_Hz",
    "band_0_400Hz",
    "band_400_800Hz",
    "band_800_1600Hz",
    "band_1600_4000Hz",
]


def extract_features(seg: np.ndarray,
                     sr: float,
                     template: np.ndarray,
                     n_fft: int = 256,
                     hop_length: int = 128) -> np.ndarray:
    """
    Compute hand-crafted features for a single 1D segment.
    """
    seg = normalize_signal(seg)

    # Time-domain features
    rms = float(np.sqrt(np.mean(seg ** 2)))
    energy = float(np.mean(seg ** 2))
    # zero-crossing rate per sample (not per second)
    zc = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0)

    # Normalized correlation with template
    try:
        ncc = normalized_cross_correlation(template, seg)
        max_corr = float(np.max(ncc))
    except Exception:
        max_corr = 0.0

    # Spectral features
    spec = compute_log_spectrogram(seg, n_fft=n_fft, hop_length=hop_length)
    spec_mean = np.mean(spec, axis=1)  # average over time
    freqs = np.linspace(0.0, sr / 2.0, spec.shape[0])

    # Shift spectrum to positive for centroid calculation
    shifted = spec_mean - np.min(spec_mean) + 1e-6
    spectral_centroid = float(np.sum(freqs * shifted) / np.sum(shifted))

    # Band-wise average log-power
    bands = [(0, 400), (400, 800), (800, 1600), (1600, 4000)]
    band_feats = []
    for fmin, fmax in bands:
        mask = (freqs >= fmin) & (freqs < fmax)
        if np.any(mask):
            band_feats.append(float(np.mean(spec_mean[mask])))
        else:
            band_feats.append(0.0)

    feats = np.array(
        [max_corr, rms, energy, zc, spectral_centroid] + band_feats,
        dtype=np.float32
    )
    return feats


def build_segment_feature_dataset(signals: List[np.ndarray],
                                  labels: np.ndarray,
                                  sr: float,
                                  template: np.ndarray,
                                  red_center_by_idx: Dict[int, int],
                                  sent_indices: np.ndarray,
                                  seg_len: int,
                                  pos_jitter_sec: float = 0.015,
                                  num_pos_per_red: int = 3,
                                  num_neg_per_sentence: int = 6
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a dataset of (features, label, sentence_id) for classical ML.

    For each RED sentence:
        - Use NCC-based centre as anchor.
        - Sample 'num_pos_per_red' positive segments with small jitter.
        - Sample ~num_neg_per_sentence/2 negative segments far from RED.

    For each non-RED sentence:
        - Sample 'num_neg_per_sentence' random segments as negative.

    Returns:
        X: (n_segments, n_features)
        y: (n_segments,) 0/1 labels
        sentence_ids: (n_segments,) sentence index (in original list)
    """
    feats_list: List[np.ndarray] = []
    y_list: List[int] = []
    sent_id_list: List[int] = []

    jitter_samples = int(pos_jitter_sec * sr)

    for idx in sent_indices:
        x = normalize_signal(signals[idx])
        L = len(x)

        # guard for very short sentences
        if L < seg_len:
            continue

        if labels[idx] == 1:
            centre_base = red_center_by_idx.get(idx, L // 2)

            # positive segments (around RED)
            for _ in range(num_pos_per_red):
                delta = 0
                if jitter_samples > 0:
                    delta = np.random.randint(-jitter_samples, jitter_samples + 1)
                centre = int(np.clip(centre_base + delta, seg_len // 2, L - seg_len // 2 - 1))
                seg = extract_segment_around_center(x, centre, seg_len)
                feats_list.append(extract_features(seg, sr, template))
                y_list.append(1)
                sent_id_list.append(idx)

            # negatives in same RED sentence, far from RED anchor
            count_neg = max(1, num_neg_per_sentence // 2)
            for _ in range(count_neg):
                for _try in range(25):
                    centre = np.random.randint(seg_len // 2, L - seg_len // 2)
                    if abs(centre - centre_base) > seg_len:
                        seg = extract_segment_around_center(x, centre, seg_len)
                        feats_list.append(extract_features(seg, sr, template))
                        y_list.append(0)
                        sent_id_list.append(idx)
                        break

        else:
            # non-RED sentence: all negatives
            for _ in range(num_neg_per_sentence):
                if L <= seg_len:
                    centre = L // 2
                else:
                    centre = np.random.randint(seg_len // 2, L - seg_len // 2)
                seg = extract_segment_around_center(x, centre, seg_len)
                feats_list.append(extract_features(seg, sr, template))
                y_list.append(0)
                sent_id_list.append(idx)

    X = np.stack(feats_list, axis=0)
    y = np.array(y_list, dtype=int)
    sentence_ids = np.array(sent_id_list, dtype=int)
    return X, y, sentence_ids


# ============================================================
# Plotting helpers
# ============================================================

def plot_template_waveform_and_spectrogram(template: np.ndarray,
                                           sr: float,
                                           plots_dir: str,
                                           prefix: str) -> None:
    ensure_dir(plots_dir)
    t = np.arange(len(template)) / sr

    # waveform
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, template, linewidth=1.0)
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
        extent=[0, spec.shape[1] * 128 / sr, 0, sr / 2.0]
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

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_example_segments_spectrograms.png"), dpi=300)
    plt.close(fig)


def plot_feature_importances(model: RandomForestClassifier,
                             plots_dir: str) -> None:
    ensure_dir(plots_dir)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    imp_sorted = importances[order]
    names_sorted = [FEATURE_NAMES[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(imp_sorted)), imp_sorted)
    ax.set_xticks(range(len(imp_sorted)))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.set_ylabel("Importance (Random Forest)")
    ax.set_title("Segment-level feature importances")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage2_feature_importances.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix_percent(cm: np.ndarray,
                                  class_names: List[str],
                                  outfile: str,
                                  title: str) -> None:
    """
    Plot confusion matrix with *percentages* per true class.
    """
    cm = np.asarray(cm, dtype=np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_percent, vmin=0, vmax=100, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_percent[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=10)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ============================================================
# Sliding window prediction and analysis
# ============================================================

def sliding_window_probs_ml(model: RandomForestClassifier,
                            signal: np.ndarray,
                            sr: float,
                            seg_len: int,
                            template: np.ndarray,
                            stride_sec: float = 0.02
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window across the signal and compute p(RED)
    for each window using the trained Random Forest.

    Returns:
        times: centre times (s) for each window
        probs: predicted probabilities (0..1)
    """
    x = normalize_signal(signal)
    L = len(x)
    stride = max(1, int(stride_sec * sr))

    times: List[float] = []
    probs: List[float] = []

    for start in range(0, max(1, L - seg_len + 1), stride):
        end = start + seg_len
        if end > L:
            seg = np.pad(x[start:], (0, end - L), mode='constant')
        else:
            seg = x[start:end]

        feats = extract_features(seg, sr, template).reshape(1, -1)
        p = float(model.predict_proba(feats)[0, 1])

        centre = start + seg_len / 2.0
        times.append(centre / sr)
        probs.append(p)

    return np.array(times), np.array(probs)


def plot_detection_examples_ml(signals: List[np.ndarray],
                               labels: np.ndarray,
                               metas: List[Dict],
                               indices: np.ndarray,
                               model: RandomForestClassifier,
                               sr: float,
                               seg_len: int,
                               template: np.ndarray,
                               plots_dir: str,
                               max_red: int = 2,
                               max_notred: int = 2,
                               threshold: float = 0.5) -> None:
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
        times, probs = sliding_window_probs_ml(model, x, sr, seg_len, template,
                                               stride_sec=0.02)
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


def plot_maxprob_hist_ml(signals: List[np.ndarray],
                         labels: np.ndarray,
                         indices: np.ndarray,
                         model: RandomForestClassifier,
                         sr: float,
                         seg_len: int,
                         template: np.ndarray,
                         plots_dir: str) -> Tuple[np.ndarray, np.ndarray]:
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
        _, probs = sliding_window_probs_ml(model, x, sr, seg_len, template)
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
    ax.set_title("Separability of RED vs non-RED by RF max probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_maxprob_hist_red_vs_notred.png"), dpi=300)
    plt.close(fig)

    return np.array(max_red), np.array(max_not)


def plot_overlay_detected_segments_ml(signals: List[np.ndarray],
                                      labels: np.ndarray,
                                      indices: np.ndarray,
                                      model: RandomForestClassifier,
                                      sr: float,
                                      seg_len: int,
                                      template: np.ndarray,
                                      plots_dir: str,
                                      threshold: float) -> None:
    """
    For each RED test sentence, take the window with highest p(RED) and
    overlay these segments (waveform).
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_indices = [i for i in indices if labels[i] == 1]
    if not red_indices:
        return

    segments = []

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        times, probs = sliding_window_probs_ml(model, x, sr, seg_len, template)
        best = int(np.argmax(probs))
        centre_time = times[best]
        centre_idx = int(centre_time * sr)
        seg = extract_segment_around_center(x, centre_idx, seg_len)
        segments.append(seg)

    seg_arr = np.stack(segments, axis=0)
    mean_seg = np.mean(seg_arr, axis=0)
    std_seg = np.std(seg_arr, axis=0)
    tt = np.arange(seg_len) / sr

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in seg_arr:
        ax.plot(tt, s, color="gray", alpha=0.3, linewidth=0.7)
    ax.plot(tt, mean_seg, color="blue", linewidth=2.0, label="Mean detected RED segment")
    ax.fill_between(tt, mean_seg - std_seg, mean_seg + std_seg,
                    color="blue", alpha=0.2, label="Mean ± 1 std")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title("RF-detected RED segments across test sentences")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_overlay_detected_red_segments_waveform.png"),
                dpi=300)
    plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(red_zip: str,
                 not_red_zip: str,
                 plots_dir: str) -> None:
    np.random.seed(0)
    ensure_dir(plots_dir)

    print("Loading dataset...")
    signals, labels, metas, sr = load_sentence_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ≈ {sr:.1f} Hz).")

    # Normalize & crop to common length
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped all signals to length {sig_len} samples (~{sig_len/sr:.2f} s).")

    # Sentence-level train/test split
    idx_all = np.arange(len(signals))
    idx_train_sent, idx_test_sent = train_test_split(
        idx_all,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    print(f"Sentence-level split: train={len(idx_train_sent)}, test={len(idx_test_sent)}")

    # --------------------------------------------------------
    # Stage 1: NCC-based initial + refined template, RED centres
    # --------------------------------------------------------
    print("Stage 1: Building initial and refined RED templates...")
    initial_template = build_initial_template(signals, metas, sr, window_duration=0.30)
    plot_template_waveform_and_spectrogram(initial_template, sr, plots_dir,
                                           prefix="Stage1_Initial_RED_template")

    refined_template, red_center_by_idx, red_segments = refine_template_and_get_centers(
        signals, labels, sr, initial_template
    )
    plot_template_waveform_and_spectrogram(refined_template, sr, plots_dir,
                                           prefix="Stage1_Refined_RED_template")

    seg_len = len(refined_template)
    print(f"Template / segment length: {seg_len} samples (~{seg_len/sr:.3f} s).")

    # --------------------------------------------------------
    # Stage 2: Build segment feature dataset & train RF
    # --------------------------------------------------------
    print("Stage 2: Building segment dataset for Random Forest...")
    X_all, y_all, sent_ids_all = build_segment_feature_dataset(
        signals=signals,
        labels=labels,
        sr=sr,
        template=refined_template,
        red_center_by_idx=red_center_by_idx,
        sent_indices=idx_train_sent,
        seg_len=seg_len,
        pos_jitter_sec=0.015,
        num_pos_per_red=3,
        num_neg_per_sentence=6
    )
    print(f"Created {len(X_all)} segments for training/validation.")
    print(f"Class balance (segments): RED={y_all.sum()}, non-RED={len(y_all)-y_all.sum()}")

    # For spectrogram visualisation
    pos_segments = []
    neg_segments = []
    for idx in range(len(X_all)):
        # we need corresponding signal segment again just for plotting;
        # simplest: regenerate from sentence id + centre from feature extraction
        # but to keep code compact we just skip detailed reconstruction here
        pass  # (optional: you can remove this pass and add plotting if needed)

    # Instead, quickly create some example segments directly
    # from a few RED / non-RED training sentences:
    ex_pos, ex_neg = [], []
    for sidx in idx_train_sent:
        x = normalize_signal(signals[sidx])
        L = len(x)
        if labels[sidx] == 1 and sidx in red_center_by_idx and len(ex_pos) < 4:
            centre = red_center_by_idx[sidx]
            ex_pos.append(extract_segment_around_center(x, centre, seg_len))
        elif labels[sidx] == 0 and len(ex_neg) < 4:
            if L <= seg_len:
                centre = L // 2
            else:
                centre = np.random.randint(seg_len // 2, L - seg_len // 2)
            ex_neg.append(extract_segment_around_center(x, centre, seg_len))
    plot_example_segments_spectrograms(ex_pos, ex_neg, sr, plots_dir)

    # Segment-level train/validation split for threshold tuning
    idx_seg_all = np.arange(len(X_all))
    idx_seg_train, idx_seg_val = train_test_split(
        idx_seg_all,
        test_size=0.2,
        stratify=y_all,
        random_state=0
    )

    X_train, y_train = X_all[idx_seg_train], y_all[idx_seg_train]
    X_val, y_val = X_all[idx_seg_val], y_all[idx_seg_val]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest trained on segment features.")

    # Segment-level validation metrics and threshold tuning
    val_probs = rf.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_val_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    print(f"Best validation F1={best_f1:.3f} at segment-level threshold={best_thr:.2f}")

    # Segment-level confusion matrix on validation set
    y_val_pred_best = (val_probs >= best_thr).astype(int)
    cm_seg = confusion_matrix(y_val, y_val_pred_best)
    print("\nSegment-level validation performance:")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_seg)
    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred_best, digits=3))
    try:
        auc_seg = roc_auc_score(y_val, val_probs)
        print(f"ROC-AUC (segment-level): {auc_seg:.3f}")
    except Exception as e:
        print("ROC-AUC (segment-level) could not be computed:", e)

    plot_feature_importances(rf, plots_dir)
    plot_confusion_matrix_percent(
        cm_seg,
        class_names=["non-RED", "RED"],
        outfile=os.path.join(plots_dir, "stage2_segment_confusion_matrix_percent.png"),
        title="Segment-level confusion matrix (%)"
    )

    # --------------------------------------------------------
    # Stage 3: Sentence-level dynamic detection on test set
    # --------------------------------------------------------
    print("\nStage 3: Evaluating dynamic RED detection on test sentences...")

    y_true_sent = labels[idx_test_sent]
    y_score_sent = []
    y_pred_sent = []

    for idx in idx_test_sent:
        x = signals[idx]
        _, probs = sliding_window_probs_ml(rf, x, sr, seg_len, refined_template,
                                           stride_sec=0.02)
        max_prob = float(np.max(probs))
        y_score_sent.append(max_prob)
        y_pred_sent.append(1 if max_prob >= best_thr else 0)

    y_score_sent = np.array(y_score_sent)
    y_pred_sent = np.array(y_pred_sent)

    cm_sent = confusion_matrix(y_true_sent, y_pred_sent)
    print("\nSentence-level performance (RF, max p(RED) with tuned threshold):")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_sent)
    print("\nClassification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))
    try:
        auc_sent = roc_auc_score(y_true_sent, y_score_sent)
        print(f"ROC-AUC (sentence-level, max p(RED)): {auc_sent:.3f}")
    except Exception as e:
        print("ROC-AUC (sentence-level) could not be computed:", e)

    plot_confusion_matrix_percent(
        cm_sent,
        class_names=["non-RED", "RED"],
        outfile=os.path.join(plots_dir, "stage3_sentence_confusion_matrix_percent.png"),
        title="Sentence-level confusion matrix (%)"
    )

    # Plots for dynamic behaviour & stability
    plot_detection_examples_ml(signals, labels, metas, idx_test_sent,
                               rf, sr, seg_len, refined_template,
                               plots_dir, threshold=best_thr)
    max_red, max_not = plot_maxprob_hist_ml(signals, labels, idx_test_sent,
                                            rf, sr, seg_len, refined_template,
                                            plots_dir)
    plot_overlay_detected_segments_ml(signals, labels, idx_test_sent,
                                      rf, sr, seg_len, refined_template,
                                      plots_dir, threshold=best_thr)

    print(f"\nAll plots saved under: {os.path.abspath(plots_dir)}")
    print("Pipeline complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML-based keyword spotting for the word 'RED' in piezoelectric signals."
    )
    parser.add_argument('--red_zip', type=str, default='Red_story_231125.zip',
                        help='Path to zip file with RED sentences.')
    parser.add_argument('--not_red_zip', type=str, default='not_RED_speaking.zip',
                        help='Path to zip file with non-RED sentences.')
    parser.add_argument('--plots_dir', type=str, default='plots_red_ml',
                        help='Directory to save plots.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.red_zip, args.not_red_zip, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
