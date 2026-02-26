#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_rf_env.py

Keyword spotting for the word "RED" in piezoelectric acoustic recordings
using a classical ML + clustering approach (no neural nets).

Goals:
------
1) Decide if a sentence contains the word "RED".
2) If yes, estimate how many times "RED" occurs in that sentence.
3) For each occurrence, estimate start/end time in seconds.

Key ideas:
----------
- Use single-word "RED" recordings in zip archives:
      "5 times Red.zip"
      "Red 100x 10Hz.zip"
      "Red 100x 13Hz.zip"
  to learn the acoustic pattern of a RED pulse:
      * estimate typical RED duration,
      * extract many positive RED segments.

- Use a Random Forest on handcrafted time-frequency features to
  distinguish RED segments vs non-RED segments.

- At test time:
      * slide a fixed-length window (~ RED duration) across each sentence,
      * compute p(RED | window) with RF,
      * cluster consecutive high-probability windows into events,
      * each event = one RED occurrence (with start/end time and confidence).

Expected files in working directory:
------------------------------------
- "Red_story_231125.zip"          # sentences containing RED
- "not_RED_speaking.zip"          # sentences without RED
- "5 times Red.zip"               # single-word RED set
- "Red 100x 10Hz.zip"             # single-word RED set
- "Red 100x 13Hz.zip"             # single-word RED set

Outputs:
--------
- Console:
    * RED pulse duration statistics
    * RF segment-level validation performance
    * sentence-level confusion matrix (has RED or not)
    * per-sentence predicted RED count and event intervals

- Plots (in "plots_red_rf_cluster/"):
    * example_detection_*.png : waveform + p(RED|t) + shaded RED events
"""

import os
import zipfile
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Matplotlib (non-interactive backend for Windows / VS Code)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Basic IO
# ----------------------------------------------------------------------

def read_csv_signal(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a stand-alone CSV file of the same format as in the zip:
    - ';' as separator
    - ',' as decimal
    - 3 header lines, then columns: Time (s), Current (nA)
    (Not used in the new pipeline, but kept for completeness.)
    """
    df = pd.read_csv(
        path,
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


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from a zip archive and return (time, current).
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
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Load main sentence-level dataset (RED / non-RED stories)
# ----------------------------------------------------------------------

def load_sentences(red_zip: str,
                   not_red_zip: str
                   ) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[str, Any]], float]:
    """
    Load all sentence-level recordings from the two zip files.

    Returns:
        signals: list of 1D arrays (current)
        labels:  np.array of ints (1 for RED sentence, 0 for non-RED)
        metas:   list of dicts (zip name, path in zip, sentence name)
        sr:      sampling rate (Hz)
    """
    signals: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[Dict[str, Any]] = []

    def _load_one(zip_path: str, label: int):
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

    return signals, np.array(labels, dtype=int), metas, sr


def crop_signals_to_min_length(signals: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    min_len = min(len(s) for s in signals)
    signals_c = [s[:min_len] for s in signals]
    return signals_c, min_len


# ----------------------------------------------------------------------
# Utilities for pulse detection in single-word RED recordings
# ----------------------------------------------------------------------

def compute_envelope(x: np.ndarray, sr: float, smooth_ms: float = 10.0) -> np.ndarray:
    """
    Compute a simple smoothed energy envelope: moving average on |x|.
    """
    x = normalize_signal(x)
    k = max(1, int(sr * smooth_ms / 1000.0))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    env = np.convolve(np.abs(x), kernel, mode='same')
    return env


def find_peaks_simple(env: np.ndarray,
                      sr: float,
                      min_distance_sec: float,
                      height_quantile: float = 0.9) -> np.ndarray:
    """
    Very simple 1D peak finder (no SciPy).
    - env: 1D array
    - min_distance_sec: minimum spacing between peaks
    - height_quantile: threshold = that quantile of env
    Returns: array of peak indices
    """
    N = len(env)
    min_dist = max(1, int(min_distance_sec * sr))
    thr = float(np.quantile(env, height_quantile))
    peaks = []
    last_idx = -min_dist
    for i in range(1, N - 1):
        if i - last_idx < min_dist:
            continue
        if env[i] > thr and env[i] > env[i - 1] and env[i] > env[i + 1]:
            peaks.append(i)
            last_idx = i
    return np.array(peaks, dtype=int)


def estimate_pulse_durations(env: np.ndarray,
                             peaks: np.ndarray,
                             sr: float,
                             rel_level: float = 0.5) -> List[float]:
    """
    For each peak, estimate duration as width where env > rel_level * peak_height.
    Returns a list of durations in seconds.
    """
    durations = []
    N = len(env)
    for p in peaks:
        h = env[p]
        thr = h * rel_level
        # go left
        i = p
        while i > 0 and env[i] > thr:
            i -= 1
        left = i
        # go right
        i = p
        while i < N - 1 and env[i] > thr:
            i += 1
        right = i
        durations.append((right - left) / sr)
    return durations


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------

def extract_features(signal: np.ndarray,
                     sr: float,
                     n_bands: int = 32) -> np.ndarray:
    """
    Extract simple time-domain and spectral features from a fixed-length segment.
    """
    x = normalize_signal(signal)
    feats: List[float] = []

    # Time-domain stats
    feats.append(float(np.mean(x)))
    feats.append(float(np.std(x)))
    feats.append(float(np.max(x)))
    feats.append(float(np.min(x)))
    feats.append(float(np.ptp(x)))
    feats.append(float(np.sqrt(np.mean(x ** 2))))  # RMS

    # Absolute value stats
    abs_x = np.abs(x)
    feats.append(float(np.mean(abs_x)))
    feats.append(float(np.std(abs_x)))
    feats.append(float(np.max(abs_x)))

    # First difference
    dx = np.diff(x)
    if len(dx) < 2:
        dx = np.zeros_like(x)
    feats.append(float(np.mean(dx)))
    feats.append(float(np.std(dx)))
    feats.append(float(np.max(dx)))
    feats.append(float(np.min(dx)))

    # Frequency-domain (binned magnitude spectrum)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mag = np.abs(np.fft.rfft(x))
    total = np.sum(mag) + 1e-12
    mag_norm = mag / total

    max_f = sr / 2.0
    edges = np.linspace(0.0, max_f, n_bands + 1)
    for i in range(n_bands):
        f0, f1 = edges[i], edges[i + 1]
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        e = float(np.sum(mag_norm[idx])) if len(idx) > 0 else 0.0
        feats.append(e)

    return np.array(feats, dtype=np.float32)


# ----------------------------------------------------------------------
# Template-based helper for RED segments inside sentences
# ----------------------------------------------------------------------

def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation (zero-mean template).
    Returns array of length len(signal) - len(template) + 1.
    """
    t = template.astype(np.float32)
    x = signal.astype(np.float32)
    L = len(t)
    if L < 1 or len(x) < L:
        raise ValueError("Template must be shorter than signal.")

    t = t - np.mean(t)
    n_t = np.linalg.norm(t)
    if n_t < 1e-8:
        raise ValueError("Template norm too small.")

    # raw correlation
    c = np.correlate(x, t, mode='valid')

    # sliding std of signal
    x2 = x ** 2
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csum2 = np.concatenate(([0.0], np.cumsum(x2)))
    sum_x = csum[L:] - csum[:-L]
    sum_x2 = csum2[L:] - csum2[:-L]
    var = sum_x2 - (sum_x ** 2) / float(L)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    ncc = c / (n_t * std)
    return ncc.astype(np.float32)


def extract_segment_around_center(signal: np.ndarray,
                                  center_idx: int,
                                  seg_len: int) -> np.ndarray:
    """
    Extract fixed-length segment around center index; zero-pad at edges if needed.
    """
    x = signal.astype(np.float32)
    half = seg_len // 2
    start = center_idx - half
    end = center_idx + half
    pad_left = pad_right = 0

    if start < 0:
        pad_left = -start
        start = 0
    if end > len(x):
        pad_right = end - len(x)
        end = len(x)

    seg = x[start:end]
    if pad_left > 0 or pad_right > 0:
        seg = np.pad(seg, (pad_left, pad_right), mode='constant')
    return seg


# ----------------------------------------------------------------------
# Single-word RED file discovery (ZIP-based, for your case)
# ----------------------------------------------------------------------

def find_single_red_files() -> List[Tuple[str, str]]:
    """
    Find single-word RED recordings inside zip archives.

    We search for zip archives whose names contain:
        "5 times red", "red 100x 10hz", "red 100x 13hz"  (case-insensitive)
    and return all their CSV members as (zip_path, member_name) pairs.
    """
    target_zip_keywords = [
        '5 times red',
        'red 100x 10hz',
        'red 100x 13hz',
    ]

    zip_paths: List[str] = []
    for fname in os.listdir('.'):
        if not fname.lower().endswith('.zip'):
            continue
        lower = fname.lower()
        if any(k in lower for k in target_zip_keywords):
            zip_paths.append(fname)

    members: List[Tuple[str, str]] = []

    for zp in zip_paths:
        with zipfile.ZipFile(zp, 'r') as z:
            for name in z.namelist():
                if name.lower().endswith('.csv'):
                    members.append((zp, name))

    return sorted(members)


# ----------------------------------------------------------------------
# Training set construction from single-word RED zips
# ----------------------------------------------------------------------

def build_red_segments_from_single_files(
    files: List[Tuple[str, str]],
    sr_assumed: float
) -> Tuple[List[np.ndarray], float, float, float]:
    """
    From multiple 'single-word RED' zip members, extract many RED segments and
    estimate RED pulse duration statistics.

    files: list of (zip_path, member_name) pairs.

    Returns:
        red_segments: list of 1D arrays (segments, NOT normalized)
        mean_dur, min_dur, max_dur: duration stats from envelopes (seconds)
    """
    if not files:
        raise RuntimeError("No single-word RED zip members provided.")

    all_durations: List[float] = []

    # Pass 1: collect pulse durations
    for zip_path, member in files:
        t, x = read_signal_from_zip(zip_path, member)
        dt = np.mean(np.diff(t))
        sr_local = 1.0 / dt
        env = compute_envelope(x, sr_local)
        peaks = find_peaks_simple(env, sr_local,
                                  min_distance_sec=0.07,
                                  height_quantile=0.9)
        if len(peaks) == 0:
            continue
        durs = estimate_pulse_durations(env, peaks, sr_local, rel_level=0.5)
        all_durations.extend(durs)

    if not all_durations:
        raise RuntimeError("No RED pulses found in single-word RED zip files.")

    all_durations = np.array(all_durations, dtype=np.float32)
    mean_dur = float(np.mean(all_durations))
    min_dur = float(np.min(all_durations))
    max_dur = float(np.max(all_durations))

    print("Estimated RED pulse durations from single-word data:")
    print(f"  mean = {mean_dur:.3f} s,  min = {min_dur:.3f} s,  max = {max_dur:.3f} s")

    # Fixed window length for segments: ~1.5 × mean duration
    seg_dur = 1.5 * mean_dur
    seg_len = int(seg_dur * sr_assumed)
    print(f"Using segment/window length ≈ {seg_dur:.3f} s ({seg_len} samples).")

    # Pass 2: extract segments around peaks
    red_segments: List[np.ndarray] = []
    for zip_path, member in files:
        t, x = read_signal_from_zip(zip_path, member)
        dt = np.mean(np.diff(t))
        sr_local = 1.0 / dt
        env = compute_envelope(x, sr_local)
        peaks = find_peaks_simple(env, sr_local,
                                  min_distance_sec=0.07,
                                  height_quantile=0.9)
        x_norm = normalize_signal(x)
        for p in peaks:
            seg = extract_segment_around_center(x_norm, p, seg_len)
            red_segments.append(seg)

    print(f"Extracted {len(red_segments)} RED segments from {len(files)} single-word RED files.")
    return red_segments, mean_dur, min_dur, max_dur


def build_training_sets(
    red_segments_single: List[np.ndarray],
    sentence_signals: List[np.ndarray],
    sentence_labels: np.ndarray,
    sr: float,
    mean_pulse_dur: float,
    seg_len: int,
    max_pos_from_sentences: int = 50,
    neg_per_nonred_sentence: int = 10
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, RandomForestClassifier]:
    """
    Build RF training data (segments) from:
      - positive segments: many from single RED recordings, plus some from RED sentences
      - negative segments: from non-RED sentences

    Train RF + StandardScaler and return them along with X, y (for inspection).
    """
    np.random.seed(0)

    # Positive segments from single RED recordings
    pos_segments = list(red_segments_single)

    # Build simple RED template from these and use it to grab segments inside RED sentences
    if len(pos_segments) > 0:
        template = np.mean(
            np.stack(pos_segments[:min(50, len(pos_segments))], axis=0),
            axis=0
        )
        template = normalize_signal(template)
    else:
        template = None

    # Add a few positive segments from RED sentences using NCC
    if template is not None:
        red_indices = np.where(sentence_labels == 1)[0]
        added = 0
        for idx in red_indices:
            if added >= max_pos_from_sentences:
                break
            x = normalize_signal(sentence_signals[idx])
            ncc = normalized_cross_correlation(template, x)
            peak = int(np.argmax(ncc))
            center = peak + len(template) // 2
            seg = extract_segment_around_center(x, center, seg_len)
            pos_segments.append(seg)
            added += 1
        print(f"Added {added} RED segments from RED sentences using NCC.")

    # Negative segments from non-RED sentences
    neg_segments: List[np.ndarray] = []
    nonred_indices = np.where(sentence_labels == 0)[0]
    for idx in nonred_indices:
        x = normalize_signal(sentence_signals[idx])
        L = len(x)
        if L < seg_len + 1:
            continue
        for _ in range(neg_per_nonred_sentence):
            start = np.random.randint(0, L - seg_len)
            seg = x[start:start + seg_len]
            neg_segments.append(seg)

    print(f"Collected {len(pos_segments)} positive and {len(neg_segments)} negative segments.")

    # Build feature matrix
    X_pos = np.stack([extract_features(s, sr) for s in pos_segments], axis=0)
    X_neg = np.stack([extract_features(s, sr) for s in neg_segments], axis=0)
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # Segment-level train/validation split
    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    # Inspect performance on segment validation set
    y_val_prob = rf.predict_proba(X_val_s)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    print("\n=== Segment-level RF performance (validation) ===")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_val_pred))
    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred, digits=3))

    return X, y, scaler, rf


# ----------------------------------------------------------------------
# Sliding window + clustering over full sentences
# ----------------------------------------------------------------------

def sliding_window_probs_rf(
    signal: np.ndarray,
    sr: float,
    seg_len: int,
    rf: RandomForestClassifier,
    scaler: StandardScaler,
    stride_sec: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window over signal, compute RF p(RED | window).
    Returns:
        centers_time: array of window center times (s)
        probs: array of probabilities
    """
    x = normalize_signal(signal)
    L = len(x)
    stride = max(1, int(stride_sec * sr))
    centers = []
    probs = []

    for start in range(0, max(1, L - seg_len + 1), stride):
        end = start + seg_len
        if end > L:
            seg = np.pad(x[start:], (0, end - L), mode='constant')
        else:
            seg = x[start:end]

        feat = extract_features(seg, sr).reshape(1, -1)
        feat_s = scaler.transform(feat)
        p = rf.predict_proba(feat_s)[0, 1]

        center = start + seg_len / 2.0
        centers.append(center / sr)
        probs.append(p)

    return np.array(centers, dtype=np.float32), np.array(probs, dtype=np.float32)


def cluster_high_prob_windows(
    centers_t: np.ndarray,
    probs: np.ndarray,
    seg_len: int,
    sr: float,
    seg_threshold: float,
    min_pulse_dur: float,
    max_pulse_dur: float,
    max_gap_sec: float = 0.03
) -> List[Dict[str, Any]]:
    """
    Cluster consecutive high-probability windows into RED events.

    Returns list of clusters, each dict containing:
        - start_s, end_s, duration_s
        - mean_prob, max_prob
        - n_windows
    """
    events: List[Dict[str, Any]] = []
    cand = np.where(probs >= seg_threshold)[0]
    if len(cand) == 0:
        return events

    max_gap = max_gap_sec
    current_cluster = [cand[0]]
    for idx in cand[1:]:
        if centers_t[idx] - centers_t[current_cluster[-1]] <= max_gap:
            current_cluster.append(idx)
        else:
            events.append(current_cluster)
            current_cluster = [idx]
    events.append(current_cluster)

    out_clusters: List[Dict[str, Any]] = []
    for cl in events:
        c_times = centers_t[cl]
        c_probs = probs[cl]
        start_s = float(c_times[0] - (seg_len / 2.0) / sr)
        end_s = float(c_times[-1] + (seg_len / 2.0) / sr)
        if start_s < 0:
            start_s = 0.0
        duration = end_s - start_s

        # Duration gating with respect to pulse duration stats
        min_allowed = max(0.5 * min_pulse_dur, 0.05)
        max_allowed = 1.8 * max_pulse_dur
        if duration < min_allowed or duration > max_allowed:
            continue

        out_clusters.append({
            'start_s': start_s,
            'end_s': end_s,
            'duration_s': duration,
            'mean_prob': float(np.mean(c_probs)),
            'max_prob': float(np.max(c_probs)),
            'n_windows': len(cl)
        })

    return out_clusters


# ----------------------------------------------------------------------
# Plotting for a few example sentences
# ----------------------------------------------------------------------

def plot_detection_example(
    signal: np.ndarray,
    sr: float,
    label: int,
    meta: Dict[str, Any],
    centers_t: np.ndarray,
    probs: np.ndarray,
    clusters: List[Dict[str, Any]],
    out_path: str,
    seg_threshold: float
) -> None:
    t = np.arange(len(signal)) / sr
    x = normalize_signal(signal)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    # Waveform
    ax1.plot(t, x, linewidth=0.7)
    for cl in clusters:
        ax1.axvspan(cl['start_s'], cl['end_s'], color='orange', alpha=0.3)
    ax1.set_ylabel("Amplitude (norm.)")
    ax1.set_title(f"Waveform | Sentence: {meta.get('sentence', 'UNKNOWN')} | True RED={label}")

    # Probability curve
    ax2.plot(centers_t, probs, linewidth=1.0)
    ax2.axhline(seg_threshold, color='red', linestyle='--', label=f"Threshold={seg_threshold:.2f}")
    for cl in clusters:
        ax2.axvspan(cl['start_s'], cl['end_s'], color='orange', alpha=0.3)
        ax2.text((cl['start_s'] + cl['end_s']) / 2.0,
                 cl['mean_prob'] + 0.02,
                 f"{cl['mean_prob']:.2f}",
                 ha='center', va='bottom', fontsize=7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("p(RED | window)")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    red_zip = "Red_story_231125.zip"
    not_red_zip = "not_RED_speaking.zip"
    plots_dir = "plots_red_rf_cluster"
    ensure_dir(plots_dir)

    print("Loading sentence-level dataset...")
    signals, labels, metas, sr = load_sentences(red_zip, not_red_zip)
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Loaded {len(signals)} sentences (sr ≈ {sr:.1f} Hz, length ≈ {sig_len/sr:.2f} s).")
    print(f"RED sentences: {int(labels.sum())}, non-RED: {len(labels) - int(labels.sum())}")

    # ---- NEW: find single-word RED data inside zip files ----
    single_members = find_single_red_files()
    if not single_members:
        raise RuntimeError(
            "No single-word RED CSV files found.\n"
            "Expected zip archives like '5 times Red.zip', 'Red 100x 10Hz.zip', "
            "'Red 100x 13Hz.zip' in this folder, each containing CSV recordings."
        )

    print("\nDetected single-word RED zip members:")
    for zp, mem in single_members:
        print(f"  {zp} :: {mem}")

    # Build RED segments + duration stats from single-word recordings
    red_segments_single, mean_pulse_dur, min_pulse_dur, max_pulse_dur = \
        build_red_segments_from_single_files(single_members, sr_assumed=sr)
    seg_len = len(red_segments_single[0])
    print(f"Segment length used for RF: {seg_len} samples (~{seg_len/sr:.3f} s).")

    # Build training sets and train RF
    X_seg, y_seg, scaler, rf = build_training_sets(
        red_segments_single=red_segments_single,
        sentence_signals=signals,
        sentence_labels=labels,
        sr=sr,
        mean_pulse_dur=mean_pulse_dur,
        seg_len=seg_len
    )

    # Sentence-level evaluation: presence/absence of RED
    idx_all = np.arange(len(signals))
    idx_train_sent, idx_test_sent = train_test_split(
        idx_all,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    seg_threshold = 0.6  # probability threshold for windows

    y_true_sent = labels[idx_test_sent]
    y_pred_sent = []

    print("\n=== Sentence-level RED detection on test set ===")

    # Also plot a few examples
    examples_plotted = 0

    for i, idx in enumerate(idx_test_sent):
        sig = signals[idx]
        lab = labels[idx]
        meta = metas[idx]

        centers_t, probs = sliding_window_probs_rf(
            sig, sr, seg_len, rf, scaler, stride_sec=0.01
        )
        clusters = cluster_high_prob_windows(
            centers_t, probs, seg_len, sr,
            seg_threshold=seg_threshold,
            min_pulse_dur=min_pulse_dur,
            max_pulse_dur=max_pulse_dur,
            max_gap_sec=0.03
        )

        has_red_pred = int(len(clusters) > 0)
        y_pred_sent.append(has_red_pred)

        # Summary line
        times_str = ", ".join(
            [f"[{cl['start_s']:.2f}, {cl['end_s']:.2f}] (p≈{cl['mean_prob']:.2f})"
             for cl in clusters]
        ) if clusters else "None"
        print(f"Sentence {i+1:02d} | True RED={lab} | Pred has_RED={has_red_pred} "
              f"| Pred count={len(clusters)} | Events: {times_str}")

        # Plot some examples
        if examples_plotted < 6:
            out_path = os.path.join(
                plots_dir,
                f"example_detection_{i+1:02d}_RED{lab}_pred{has_red_pred}.png"
            )
            plot_detection_example(
                sig, sr, int(lab), meta,
                centers_t, probs, clusters,
                out_path,
                seg_threshold=seg_threshold
            )
            examples_plotted += 1

    y_true_sent = np.array(y_true_sent, dtype=int)
    y_pred_sent = np.array(y_pred_sent, dtype=int)

    print("\nSentence-level confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_sent, y_pred_sent))
    print("\nSentence-level classification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))

    print(f"\nPlots saved into: {os.path.abspath(plots_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()

