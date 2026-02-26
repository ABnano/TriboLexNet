#!/usr/bin/env python3
"""Core data loading, signal processing, feature extraction, and RF modeling."""

import os
import zipfile
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from .plotting import (
    plot_confusion_matrix_percent_dual,
    plot_feature_importances_dual,
    plot_pulse_duration_hist_dual,
)

def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from a zip archive and return (time, current).
    CSV format:
      - ';' separator
      - ',' decimal
      - 3 header lines
      - columns: Time (s), Current (nA)
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
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x

def load_sentences(
    red_zip: str,
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

                _, x = read_signal_from_zip(zip_path, m)
                signals.append(x)
                labels.append(label)
                metas.append({
                    "zip": os.path.basename(zip_path),
                    "path": m,
                    "sentence": sentence
                })

    _load_one(red_zip, label=1)
    _load_one(not_red_zip, label=0)

    # Estimate sampling rate from first RED file
    with zipfile.ZipFile(red_zip, "r") as z:
        first_csv = sorted([m for m in z.namelist() if m.lower().endswith(".csv")])[0]
        t, _ = read_signal_from_zip(red_zip, first_csv)
        dt = np.mean(np.diff(t))
        sr = float(1.0 / dt)

    return signals, np.array(labels, dtype=int), metas, sr

def crop_signals_to_min_length(signals: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    min_len = min(len(s) for s in signals)
    signals_c = [s[:min_len] for s in signals]
    return signals_c, min_len

def compute_envelope(x: np.ndarray, sr: float, smooth_ms: float = 10.0) -> np.ndarray:
    """
    Compute a simple smoothed envelope: moving average on |x|.
    """
    x = normalize_signal(x)
    k = max(1, int(sr * smooth_ms / 1000.0))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    env = np.convolve(np.abs(x), kernel, mode="same")
    return env

def find_peaks_simple(
    env: np.ndarray,
    sr: float,
    min_distance_sec: float,
    height_quantile: float = 0.9
) -> np.ndarray:
    """
    Very simple 1D peak finder (no SciPy).
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

def estimate_pulse_durations(
    env: np.ndarray,
    peaks: np.ndarray,
    sr: float,
    rel_level: float = 0.5
) -> List[float]:
    """
    For each peak, estimate duration as width where env > rel_level * peak_height.
    Returns durations in seconds.
    """
    durations = []
    N = len(env)

    for p in peaks:
        h = env[p]
        thr = h * rel_level

        i = p
        while i > 0 and env[i] > thr:
            i -= 1
        left = i

        i = p
        while i < N - 1 and env[i] > thr:
            i += 1
        right = i

        durations.append((right - left) / sr)

    return durations

def extract_features(signal: np.ndarray, sr: float, n_bands: int = 32) -> np.ndarray:
    """
    Extract simple time-domain + spectral features from a segment.
    """
    x = normalize_signal(signal)
    feats: List[float] = []

    # Time-domain stats
    feats.append(float(np.mean(x)))                  # 0
    feats.append(float(np.std(x)))                   # 1
    feats.append(float(np.max(x)))                   # 2
    feats.append(float(np.min(x)))                   # 3
    feats.append(float(np.ptp(x)))                   # 4
    feats.append(float(np.sqrt(np.mean(x ** 2))))    # 5 (RMS)

    # Absolute value stats
    abs_x = np.abs(x)
    feats.append(float(np.mean(abs_x)))              # 6
    feats.append(float(np.std(abs_x)))               # 7
    feats.append(float(np.max(abs_x)))               # 8

    # First difference
    dx = np.diff(x)
    if len(dx) < 2:
        dx = np.zeros_like(x)
    feats.append(float(np.mean(dx)))                 # 9
    feats.append(float(np.std(dx)))                  # 10
    feats.append(float(np.max(dx)))                  # 11
    feats.append(float(np.min(dx)))                  # 12

    # Frequency-domain (binned spectrum magnitude)
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

def get_feature_names(sr: float, n_bands: int = 32) -> List[str]:
    names: List[str] = [
        "Time mean",
        "Time std",
        "Time max",
        "Time min",
        "Time peak-to-peak",
        "Time RMS",
        "Abs mean",
        "Abs std",
        "Abs max",
        "Diff mean",
        "Diff std",
        "Diff max",
        "Diff min",
    ]
    max_f = sr / 2.0
    edges = np.linspace(0.0, max_f, n_bands + 1)
    for i in range(n_bands):
        f0, f1 = edges[i], edges[i + 1]
        names.append(f"Band {i:02d}: {int(f0)}–{int(f1)} Hz")
    return names

def normalized_cross_correlation(template: np.ndarray, signal: np.ndarray) -> np.ndarray:
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

    c = np.correlate(x, t, mode="valid")

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

def extract_segment_around_center(signal: np.ndarray, center_idx: int, seg_len: int) -> np.ndarray:
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
        seg = np.pad(seg, (pad_left, pad_right), mode="constant")
    return seg

def find_single_red_files(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find single-word RED recordings inside zip archives in input_dir.

    Looks for zip names containing:
      - "5 times red"
      - "red 100x 10hz"
      - "red 100x 13hz"
    and returns all CSV members as (zip_path, member_name).
    """
    target_zip_keywords = [
        "5 times red",
        "red 100x 10hz",
        "red 100x 13hz",
    ]

    zip_paths: List[str] = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".zip"):
            continue
        lower = fname.lower()
        if any(k in lower for k in target_zip_keywords):
            zip_paths.append(os.path.join(input_dir, fname))

    members: List[Tuple[str, str]] = []
    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    members.append((zp, name))

    return sorted(members, key=lambda x: (os.path.basename(x[0]).lower(), x[1].lower()))

def build_red_segments_from_single_files(
    files: List[Tuple[str, str]],
    sr_assumed: float,
    plots_dir: str,
    dpi: int,
    fs: Dict[str, int]
) -> Tuple[List[np.ndarray], float, float, float, int]:
    """
    Extract RED segments from single-word data + estimate pulse duration stats.

    Returns:
        red_segments, mean_dur, min_dur, max_dur, seg_len
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
        peaks = find_peaks_simple(env, sr_local, min_distance_sec=0.07, height_quantile=0.9)
        if len(peaks) == 0:
            continue

        durs = estimate_pulse_durations(env, peaks, sr_local, rel_level=0.5)
        all_durations.extend(durs)

    if not all_durations:
        raise RuntimeError("No RED pulses found in single-word RED zip files.")

    all_durations_arr = np.array(all_durations, dtype=np.float32)
    mean_dur = float(np.mean(all_durations_arr))
    min_dur = float(np.min(all_durations_arr))
    max_dur = float(np.max(all_durations_arr))

    print("Estimated RED pulse durations from single-word data:")
    print(f"  mean = {mean_dur:.3f} s,  min = {min_dur:.3f} s,  max = {max_dur:.3f} s")

    # Save pulse duration histogram (titled + notitle)
    plot_pulse_duration_hist_dual(
        all_durations_arr, mean_dur,
        out_png=os.path.join(plots_dir, "single_red_pulse_duration_hist.png"),
        dpi=dpi,
        fs=fs
    )

    # Segment length: ~1.5 × mean duration
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
        peaks = find_peaks_simple(env, sr_local, min_distance_sec=0.07, height_quantile=0.9)

        x_norm = normalize_signal(x)
        for p in peaks:
            seg = extract_segment_around_center(x_norm, p, seg_len)
            red_segments.append(seg)

    print(f"Extracted {len(red_segments)} RED segments from {len(files)} single-word RED files.")
    return red_segments, mean_dur, min_dur, max_dur, seg_len

def build_training_sets(
    red_segments_single: List[np.ndarray],
    sentence_signals: List[np.ndarray],
    sentence_labels: np.ndarray,
    sr: float,
    seg_len: int,
    plots_dir: str,
    dpi: int,
    fs: Dict[str, int],
    top_k_features: int,
    max_pos_from_sentences: int = 50,
    neg_per_nonred_sentence: int = 10
) -> Tuple[StandardScaler, RandomForestClassifier]:
    """
    Build RF training data (segments) from:
      - positives: many from single RED recordings + some from RED sentences
      - negatives: random windows from non-RED sentences
    """
    np.random.seed(0)

    pos_segments = list(red_segments_single)

    # Template from positives (to grab additional positives from RED sentences)
    if len(pos_segments) > 0:
        template = np.mean(np.stack(pos_segments[:min(50, len(pos_segments))], axis=0), axis=0)
        template = normalize_signal(template)
    else:
        template = None

    # Add a few positives from RED sentences using NCC
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

    X_pos = np.stack([extract_features(s, sr) for s in pos_segments], axis=0)
    X_neg = np.stack([extract_features(s, sr) for s in neg_segments], axis=0)
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

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
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    y_val_prob = rf.predict_proba(X_val_s)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    cm_seg = confusion_matrix(y_val, y_val_pred)

    print("\n=== Segment-level RF performance (validation) ===")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_seg)
    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred, digits=3))

    plot_confusion_matrix_percent_dual(
        cm_seg,
        class_names=["Non-RED segment", "RED segment"],
        out_png=os.path.join(plots_dir, "segment_rf_confusion_matrix_percent.png"),
        title="Segment-level RF confusion matrix",
        dpi=dpi,
        fs=fs
    )

    feature_names = get_feature_names(sr, n_bands=32)
    plot_feature_importances_dual(
        rf,
        feature_names,
        out_png=os.path.join(plots_dir, "segment_rf_feature_importances.png"),
        dpi=dpi,
        fs=fs,
        top_k=top_k_features
    )

    return scaler, rf

def sliding_window_probs_rf(
    signal: np.ndarray,
    sr: float,
    seg_len: int,
    rf: RandomForestClassifier,
    scaler: StandardScaler,
    stride_sec: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window over signal, compute RF p(RED|window).
    """
    x = normalize_signal(signal)
    L = len(x)
    stride = max(1, int(stride_sec * sr))

    centers = []
    probs = []

    for start in range(0, max(1, L - seg_len + 1), stride):
        end = start + seg_len
        if end > L:
            seg = np.pad(x[start:], (0, end - L), mode="constant")
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
    """
    cand = np.where(probs >= seg_threshold)[0]
    if len(cand) == 0:
        return []

    clusters = []
    cur = [cand[0]]
    for idx in cand[1:]:
        if centers_t[idx] - centers_t[cur[-1]] <= max_gap_sec:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)

    out_clusters: List[Dict[str, Any]] = []
    for cl in clusters:
        c_times = centers_t[cl]
        c_probs = probs[cl]

        start_s = float(c_times[0] - (seg_len / 2.0) / sr)
        end_s = float(c_times[-1] + (seg_len / 2.0) / sr)
        start_s = max(0.0, start_s)
        duration = end_s - start_s

        min_allowed = max(0.5 * min_pulse_dur, 0.05)
        max_allowed = 1.8 * max_pulse_dur
        if duration < min_allowed or duration > max_allowed:
            continue

        out_clusters.append({
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration,
            "mean_prob": float(np.mean(c_probs)),
            "max_prob": float(np.max(c_probs)),
            "n_windows": len(cl)
        })

    return out_clusters
