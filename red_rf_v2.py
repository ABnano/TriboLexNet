#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_spotting_rf_v2.py

Keyword spotting for the word "RED" in piezoelectric acoustic recordings
using a conventional ML classifier (Random Forest) + duration-aware detection.

Pipeline (high level)
---------------------
1) Sentence-level acoustic signals are read from two zip files:
      - RED sentences (contain the word RED somewhere)
      - non-RED sentences (same stories, no RED word)
2) We build an initial RED template from simple phrases
      ("RED before story", "RED after the story"),
   and then refine it via normalized cross-correlation (NCC) to obtain
   RED centers in all RED sentences (weak supervision).
3) Around each RED center we extract fixed-length segments (~0.30 s).
   From RED sentences and non-RED sentences we build a segment dataset:
      - positives: windows around the NCC-based RED centers (with jitter)
      - negatives: windows from other parts of RED sentences + all of
                   non-RED sentences.
4) For each segment, we compute spectrogram-based features:
      - log-power spectrogram
      - per-frequency mean and std
      - zero-crossing rate of the raw waveform
   → feature vector per segment (size ≈ 2*F + 1).
5) Train a RandomForest classifier (segment-level RED vs non-RED).
   We report segment-level confusion matrix and classification metrics.
6) Dynamic keyword detection:
      - Slide the same 0.30 s window across each sentence.
      - For each window, compute features → RF → p(RED | window center).
      - Smooth probabilities over time.
      - Find contiguous regions where p(RED) ≥ PROB_THRESHOLD (default 0.5).
      - Keep only regions whose duration ≥ MIN_WORD_DUR (≈0.11 s,
        tuned from the "Red 100x 10Hz/13Hz" repeated-RED recordings).
      - A sentence is labelled RED if at least one such region exists.
7) We plot:
      - Sentence-level waveform + p(RED | t) for some RED/non-RED examples.
      - Histograms of max p(RED) over sentence (RED vs non-RED).
      - Overlay of all detected RED segments (waveform).
      - Average log-power spectrogram of detected RED segments.
      - Segment- and sentence-level confusion matrices (percent).
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
)

# non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Basic IO & utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current).

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


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x /= std
    return x


def load_dataset(red_zip: str,
                 not_red_zip: str) -> Tuple[List[np.ndarray], List[int], List[Dict], float]:
    """
    Load all signals from the two zip files.

    Returns
    -------
    signals : list of 1D float32 arrays
    labels  : list of 0/1 (1 = RED sentence, 0 = non-RED)
    metas   : list of dicts with metadata (zip, path, sentence ID)
    sr      : float, sampling rate (Hz) estimated from first RED file
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

    # Estimate sampling rate from first RED file
    with zipfile.ZipFile(red_zip, 'r') as z:
        first_csv = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])[0]
        t, _ = read_signal_from_zip(red_zip, first_csv)
        dt = np.mean(np.diff(t))
        sr = float(1.0 / dt)

    return signals, labels, metas, sr


def crop_signals_to_min_length(signals: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    min_len = min(len(x) for x in signals)
    cropped = [x[:min_len] for x in signals]
    return cropped, min_len


# ---------------------------------------------------------------------
# Spectrogram & NCC utilities
# ---------------------------------------------------------------------

def compute_log_spectrogram(x: np.ndarray,
                            n_fft: int = 256,
                            hop_length: int = 128) -> np.ndarray:
    """
    Simple log-power spectrogram with Hann window.

    Returns: array of shape (n_freq, n_frames)
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
    power = np.abs(spec) ** 2
    power = power.T  # (n_freq, n_frames)
    log_spec = np.log10(power + 1e-12)
    return log_spec


def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation between template and signal (1D).
    Returns values in approx [-1, 1].
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

    c = np.correlate(x, t, mode='valid')

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
    Build an initial RED template from simple sentences.

    We find the max-energy region in those sentences and average
    a fixed-duration window around it.
    """
    win_len = int(window_duration * sr)
    segs = []

    for x, m in zip(signals, metas):
        if m.get('sentence') not in RED_TEMPLATE_SENTENCES:
            continue
        x_norm = normalize_signal(x)
        abs_x = np.abs(x_norm)
        kernel_size = max(1, int(0.02 * sr))
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
    template -= np.mean(template)
    nrm = np.linalg.norm(template)
    if nrm > 1e-8:
        template /= nrm
    return template.astype(np.float32)


def refine_template_and_get_centers(signals: List[np.ndarray],
                                    labels: np.ndarray,
                                    sr: float,
                                    initial_template: np.ndarray
                                    ) -> Tuple[np.ndarray, Dict[int, int], List[np.ndarray]]:
    """
    Use NCC with the initial template to find RED centers in all RED sentences,
    extract aligned segments, and build a refined template.
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
        raise RuntimeError("No RED segments in refinement step.")

    seg_arr = np.stack(segments, axis=0)
    refined = np.mean(seg_arr, axis=0)
    refined -= np.mean(refined)
    nrm = np.linalg.norm(refined)
    if nrm > 1e-8:
        refined /= nrm
    return refined.astype(np.float32), red_center_by_idx, segments


# ---------------------------------------------------------------------
# Feature extraction for ML classifier
# ---------------------------------------------------------------------

def segment_features(seg: np.ndarray,
                     sr: float,
                     n_fft: int = 256,
                     hop_length: int = 128) -> np.ndarray:
    """
    Turn one waveform segment into a 1D feature vector.

    - log-power spectrogram (F x T)
    - take mean and std over time for each frequency bin
    - append zero-crossing rate of waveform
    """
    seg = normalize_signal(seg)
    spec = compute_log_spectrogram(seg, n_fft=n_fft, hop_length=hop_length)
    mean_spec = np.mean(spec, axis=1)
    std_spec = np.std(spec, axis=1)
    # zero-crossing rate
    zc = np.mean(seg[:-1] * seg[1:] < 0.0)
    feats = np.concatenate([mean_spec, std_spec, np.array([zc], dtype=np.float32)], axis=0)
    return feats.astype(np.float32)


def create_segment_dataset(signals: List[np.ndarray],
                           labels: np.ndarray,
                           red_center_by_idx: Dict[int, int],
                           sr: float,
                           indices: np.ndarray,
                           seg_len: int,
                           pos_jitter_sec: float = 0.02,
                           num_pos_per_red: int = 3,
                           num_neg_per_sentence: int = 6
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build segment-level dataset.

    For RED sentences:
      - 'num_pos_per_red' positive windows around the RED center (with jitter)
      - some negatives from far away regions in same sentence
    For non-RED sentences:
      - 'num_neg_per_sentence' random windows (all negatives)
    """
    jitter_samples = int(pos_jitter_sec * sr)
    feats_list: List[np.ndarray] = []
    lab_list: List[int] = []

    for idx in indices:
        x = normalize_signal(signals[idx])
        L = len(x)
        if labels[idx] == 1:  # RED sentence
            if idx not in red_center_by_idx:
                center_base = L // 2
            else:
                center_base = red_center_by_idx[idx]

            # positives
            for _ in range(num_pos_per_red):
                delta = np.random.randint(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
                center = int(np.clip(center_base + delta, seg_len // 2, L - seg_len // 2))
                seg = extract_segment_around_center(x, center, seg_len)
                feats_list.append(segment_features(seg, sr))
                lab_list.append(1)

            # negatives (far from RED)
            for _ in range(max(1, num_neg_per_sentence // 2)):
                for _try in range(20):
                    c = np.random.randint(seg_len // 2, L - seg_len // 2)
                    if abs(c - center_base) > seg_len:
                        seg = extract_segment_around_center(x, c, seg_len)
                        feats_list.append(segment_features(seg, sr))
                        lab_list.append(0)
                        break
        else:  # non-RED sentence
            for _ in range(num_neg_per_sentence):
                c = np.random.randint(seg_len // 2, L - seg_len // 2)
                seg = extract_segment_around_center(x, c, seg_len)
                feats_list.append(segment_features(seg, sr))
                lab_list.append(0)

    X = np.stack(feats_list, axis=0)
    y = np.array(lab_list, dtype=int)
    return X, y


# ---------------------------------------------------------------------
# Sliding-window probabilities + duration-aware detection
# ---------------------------------------------------------------------

def sliding_window_features(signal: np.ndarray,
                            sr: float,
                            seg_len: int,
                            stride_sec: float = 0.015
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window over the signal and compute feature
    vectors for each window. Returns (times, feature_matrix).
    """
    x = normalize_signal(signal)
    L = len(x)
    stride = max(1, int(stride_sec * sr))

    times: List[float] = []
    feats: List[np.ndarray] = []

    for start in range(0, max(1, L - seg_len + 1), stride):
        end = start + seg_len
        if end > L:
            seg = np.pad(x[start:], (0, end - L), mode='constant')
        else:
            seg = x[start:end]
        feats.append(segment_features(seg, sr))
        center = start + seg_len / 2.0
        times.append(center / sr)

    return np.array(times), np.stack(feats, axis=0)


def smooth_probabilities(probs: np.ndarray, smooth_win: int = 5) -> np.ndarray:
    """
    Simple moving-average smoothing over time indices.
    smooth_win is the number of frames in the averaging window.
    """
    if smooth_win <= 1:
        return probs
    kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
    return np.convolve(probs, kernel, mode='same')


def detect_red_regions(times: np.ndarray,
                       probs: np.ndarray,
                       prob_threshold: float,
                       min_word_dur: float) -> List[Tuple[float, float, float]]:
    """
    Detect contiguous regions where p(RED) stays above threshold
    for at least min_word_dur seconds.

    Returns list of (start_time, end_time, peak_prob) for each region.
    """
    if len(times) == 0:
        return []

    # approx frame spacing
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
    mask = probs >= prob_threshold

    regions: List[Tuple[float, float, float]] = []
    in_reg = False
    start_idx = 0

    for i, flag in enumerate(mask):
        if flag and not in_reg:
            in_reg = True
            start_idx = i
        elif not flag and in_reg:
            in_reg = False
            end_idx = i
            dur = (end_idx - start_idx) * dt
            if dur >= min_word_dur:
                seg_probs = probs[start_idx:end_idx]
                peak = float(np.max(seg_probs))
                regions.append((times[start_idx], times[end_idx - 1], peak))
    if in_reg:
        end_idx = len(mask)
        dur = (end_idx - start_idx) * dt
        if dur >= min_word_dur:
            seg_probs = probs[start_idx:end_idx]
            peak = float(np.max(seg_probs))
            regions.append((times[start_idx], times[end_idx - 1], peak))

    return regions


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_template(template: np.ndarray,
                  sr: float,
                  plots_dir: str,
                  prefix: str) -> None:
    ensure_dir(plots_dir)
    t = np.arange(len(template)) / sr

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, template)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"{prefix} waveform")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{prefix}_waveform.png"), dpi=300)
    plt.close(fig)

    spec = compute_log_spectrogram(template)
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


def plot_confusion_matrix_percent(cm: np.ndarray,
                                  class_names: List[str],
                                  title: str,
                                  save_path: str) -> None:
    cm = cm.astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = 100.0 * cm / np.maximum(row_sums, 1e-9)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_detection_examples(signals: List[np.ndarray],
                            labels: np.ndarray,
                            metas: List[Dict],
                            indices: np.ndarray,
                            rf: RandomForestClassifier,
                            sr: float,
                            seg_len: int,
                            prob_threshold: float,
                            min_word_dur: float,
                            plots_dir: str,
                            max_red: int = 2,
                            max_notred: int = 2) -> None:
    """
    Plot waveform + p(RED|t) and detected RED regions for a few sentences.
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

        # probabilities + detected regions
        times, feats = sliding_window_features(x, sr, seg_len)
        probs = rf.predict_proba(feats)[:, 1]
        probs_s = smooth_probabilities(probs, smooth_win=5)
        regions = detect_red_regions(times, probs_s, prob_threshold, min_word_dur)

        ax_p = axes[row, 1]
        ax_p.plot(times, probs_s, linewidth=1.0)
        ax_p.axhline(prob_threshold, color="red", linestyle="--", label="Threshold")

        for (s_t, e_t, peak) in regions:
            center = 0.5 * (s_t + e_t)
            ax_p.axvspan(s_t, e_t, color="green", alpha=0.2)
            ax_p.axvline(center, color="green", linestyle="--")

        max_prob = float(np.max(probs_s)) if len(probs_s) > 0 else 0.0
        ax_p.set_title(f"p(RED | t) (max={max_prob:.3f})")
        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Probability")
        ax_p.set_ylim(0.0, 1.05)
        ax_p.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_detection_examples_rf.png"), dpi=300)
    plt.close(fig)


def plot_maxprob_hist(signals: List[np.ndarray],
                      labels: np.ndarray,
                      indices: np.ndarray,
                      rf: RandomForestClassifier,
                      sr: float,
                      seg_len: int,
                      prob_threshold: float,
                      min_word_dur: float,
                      plots_dir: str) -> None:
    """
    For each sentence, compute max p(RED) (after smoothing) and plot histograms
    for RED vs non-RED sentences.
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)

    max_red = []
    max_not = []

    for idx in indices:
        x = normalize_signal(signals[idx])
        times, feats = sliding_window_features(x, sr, seg_len)
        probs = rf.predict_proba(feats)[:, 1] if len(feats) > 0 else np.array([0.0])
        probs_s = smooth_probabilities(probs, smooth_win=5)
        regions = detect_red_regions(times, probs_s, prob_threshold, min_word_dur)

        if regions:
            peak = max(r[2] for r in regions)
        else:
            peak = float(np.max(probs_s)) if len(probs_s) > 0 else 0.0

        if labels[idx] == 1:
            max_red.append(peak)
        else:
            max_not.append(peak)

    fig, ax = plt.subplots(figsize=(6, 4))
    if max_red:
        ax.hist(max_red, bins=10, alpha=0.7, density=True, label="RED sentences")
    if max_not:
        ax.hist(max_not, bins=10, alpha=0.7, density=True, label="Non-RED sentences")
    ax.set_xlabel("Max p(RED) over sentence (after duration gating)")
    ax.set_ylabel("Density")
    ax.set_title("Separability of RED vs non-RED by RF max probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_maxprob_hist_rf.png"), dpi=300)
    plt.close(fig)


def plot_overlay_detected_segments(signals: List[np.ndarray],
                                   labels: np.ndarray,
                                   indices: np.ndarray,
                                   rf: RandomForestClassifier,
                                   sr: float,
                                   seg_len: int,
                                   prob_threshold: float,
                                   min_word_dur: float,
                                   plots_dir: str) -> None:
    """
    For each RED test sentence, take the duration-gated region with
    highest p(RED), extract that waveform segment, and overlay them.
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_indices = [i for i in indices if labels[i] == 1]
    if not red_indices:
        return

    segments = []

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        times, feats = sliding_window_features(x, sr, seg_len)
        if len(feats) == 0:
            continue
        probs = rf.predict_proba(feats)[:, 1]
        probs_s = smooth_probabilities(probs, smooth_win=5)
        regions = detect_red_regions(times, probs_s, prob_threshold, min_word_dur)
        if not regions:
            continue
        # choose the region with highest peak probability
        region = max(regions, key=lambda r: r[2])
        s_t, e_t, _ = region
        center = 0.5 * (s_t + e_t)
        center_idx = int(center * sr)
        seg = extract_segment_around_center(x, center_idx, seg_len)
        segments.append(seg)

    if not segments:
        return

    seg_arr = np.stack(segments, axis=0)
    mean_seg = np.mean(seg_arr, axis=0)
    std_seg = np.std(seg_arr, axis=0)
    tt = np.arange(seg_len) / sr

    # waveform overlay
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in seg_arr:
        ax.plot(tt, s, color="gray", alpha=0.25, linewidth=0.6)
    ax.plot(tt, mean_seg, color="blue", linewidth=2.0, label="Mean detected RED segment")
    ax.fill_between(tt, mean_seg - std_seg, mean_seg + std_seg,
                    color="blue", alpha=0.2, label="Mean ± 1 std")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title("RF-detected RED segments across test sentences")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_overlay_red_segments_rf.png"), dpi=300)
    plt.close(fig)

    # average spectrogram
    specs = []
    min_frames = None
    for s in segments:
        spec = compute_log_spectrogram(s)
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
    ax.set_title("Average log-power spectrogram of RF-detected RED segments")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "stage3_average_red_spectrogram_rf.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(red_zip: str,
                 not_red_zip: str,
                 plots_dir: str,
                 num_trees: int = 300,
                 prob_threshold: float = 0.5,
                 min_word_dur: float = 0.11) -> None:
    """
    Run the full RF-based keyword spotting pipeline.
    """
    ensure_dir(plots_dir)

    print("Loading dataset...")
    signals, labels, metas, sr = load_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ≈ {sr:.1f} Hz).")

    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped all signals to common length: {sig_len} samples (~{sig_len/sr:.2f} s).")

    labels = np.array(labels, dtype=int)

    # sentence-level split
    idx_all = np.arange(len(signals))
    idx_train_sent, idx_test_sent = train_test_split(
        idx_all,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    print(f"Sentence-level split: train={len(idx_train_sent)}, test={len(idx_test_sent)}")

    # Stage 1: template + NCC-based centers
    print("Stage 1: Building initial and refined RED templates...")
    initial_template = build_initial_template(signals, metas, sr, window_duration=0.30)
    refined_template, red_center_by_idx, red_segments = refine_template_and_get_centers(
        signals, labels, sr, initial_template
    )
    seg_len = len(refined_template)
    print(f"Template / segment length: {seg_len} samples (~{seg_len/sr:.3f} s).")

    plot_template(initial_template, sr, plots_dir, prefix="Stage1_Initial_RED_template")
    plot_template(refined_template, sr, plots_dir, prefix="Stage1_Refined_RED_template")

    # Stage 2: segment dataset + RF training
    print("Stage 2: Building segment dataset and training RF classifier...")
    X_all, y_all = create_segment_dataset(
        signals=signals,
        labels=labels,
        red_center_by_idx=red_center_by_idx,
        sr=sr,
        indices=idx_train_sent,
        seg_len=seg_len,
        pos_jitter_sec=0.02,
        num_pos_per_red=3,
        num_neg_per_sentence=6
    )
    print(f"Segment dataset size: {len(y_all)} segments.")
    print(f"Class balance: RED={y_all.sum()}, non-RED={len(y_all)-y_all.sum()}")

    # segment-level split
    idx_seg = np.arange(len(y_all))
    idx_seg_train, idx_seg_val = train_test_split(
        idx_seg,
        test_size=0.2,
        stratify=y_all,
        random_state=0
    )
    X_train, y_train = X_all[idx_seg_train], y_all[idx_seg_train]
    X_val, y_val = X_all[idx_seg_val], y_all[idx_seg_val]

    rf = RandomForestClassifier(
        n_estimators=num_trees,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=0
    )
    rf.fit(X_train, y_train)
    print("RandomForest trained.")

    # segment-level performance
    y_val_pred = rf.predict(X_val)
    y_val_proba = rf.predict_proba(X_val)[:, 1]

    print("\nSegment-level performance (validation set):")
    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, digits=3))
    try:
        auc_seg = roc_auc_score(y_val, y_val_proba)
        print(f"ROC-AUC (segment-level): {auc_seg:.3f}")
    except Exception as e:
        print("ROC-AUC (segment-level) could not be computed:", e)

    cm_seg = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix_percent(
        cm_seg,
        class_names=["non-RED", "RED"],
        title="Segment-level confusion matrix (%)",
        save_path=os.path.join(plots_dir, "stage2_segment_confusion_percent_rf.png")
    )

    # Stage 3: sentence-level dynamic detection
    print("\nStage 3: Sentence-level dynamic detection with duration gating...")
    y_true_sent = labels[idx_test_sent]
    y_pred_sent = []
    y_score_sent = []

    for idx in idx_test_sent:
        x = normalize_signal(signals[idx])
        times, feats = sliding_window_features(x, sr, seg_len)
        if len(feats) == 0:
            y_pred_sent.append(0)
            y_score_sent.append(0.0)
            continue

        probs = rf.predict_proba(feats)[:, 1]
        probs_s = smooth_probabilities(probs, smooth_win=5)
        regions = detect_red_regions(times, probs_s, prob_threshold, min_word_dur)

        if regions:
            y_pred_sent.append(1)
            peak = max(r[2] for r in regions)
        else:
            y_pred_sent.append(0)
            peak = float(np.max(probs_s))
        y_score_sent.append(peak)

    y_pred_sent = np.array(y_pred_sent, dtype=int)
    y_score_sent = np.array(y_score_sent, dtype=float)

    print("\nSentence-level performance (duration-aware RF detector):")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_sent, y_pred_sent))
    print("\nClassification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))
    try:
        auc_sent = roc_auc_score(y_true_sent, y_score_sent)
        print(f"ROC-AUC (sentence-level, using peak prob): {auc_sent:.3f}")
    except Exception as e:
        print("ROC-AUC (sentence-level) could not be computed:", e)

    cm_sent = confusion_matrix(y_true_sent, y_pred_sent)
    plot_confusion_matrix_percent(
        cm_sent,
        class_names=["non-RED", "RED"],
        title="Sentence-level confusion matrix (%)",
        save_path=os.path.join(plots_dir, "stage3_sentence_confusion_percent_rf.png")
    )

    # interpretability / diagnostics plots
    plot_detection_examples(
        signals, labels, metas, idx_test_sent, rf, sr, seg_len,
        prob_threshold=prob_threshold,
        min_word_dur=min_word_dur,
        plots_dir=plots_dir,
        max_red=2,
        max_notred=2
    )
    plot_maxprob_hist(
        signals, labels, idx_test_sent, rf, sr, seg_len,
        prob_threshold=prob_threshold,
        min_word_dur=min_word_dur,
        plots_dir=plots_dir
    )
    plot_overlay_detected_segments(
        signals, labels, idx_test_sent, rf, sr, seg_len,
        prob_threshold=prob_threshold,
        min_word_dur=min_word_dur,
        plots_dir=plots_dir
    )

    print(f"\nAll plots saved under: {os.path.abspath(plots_dir)}")
    print("RF-based RED keyword spotting pipeline complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RandomForest-based keyword spotting for the word 'RED' in piezoelectric signals."
    )
    parser.add_argument('--red_zip', type=str, default='Red_story_231125.zip',
                        help='Path to zip file with RED sentences.')
    parser.add_argument('--not_red_zip', type=str, default='not_RED_speaking.zip',
                        help='Path to zip file with non-RED sentences.')
    parser.add_argument('--plots_dir', type=str, default='plots_red_rf_v2',
                        help='Directory to save plots.')
    parser.add_argument('--trees', type=int, default=300,
                        help='Number of trees in the random forest.')
    parser.add_argument('--prob_threshold', type=float, default=0.5,
                        help='Probability threshold for duration-aware detection.')
    parser.add_argument('--min_word_dur', type=float, default=0.11,
                        help='Minimum duration (s) of a RED region.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        red_zip=args.red_zip,
        not_red_zip=args.not_red_zip,
        plots_dir=args.plots_dir,
        num_trees=args.trees,
        prob_threshold=args.prob_threshold,
        min_word_dur=args.min_word_dur
    )


if __name__ == "__main__":
    main()
