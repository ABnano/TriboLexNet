#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_detect_rf.py

Keyword spotting for the word "RED" in piezoelectric acoustic recordings
using a classical ML + clustering approach (no neural nets).

What you asked (implemented):
-----------------------------
1) Configurable input/output folders:
      --input_dir  (where your zip files are)
      --out_dir    (where results are saved)

2) High-resolution plots:
      --dpi (default 600)

3) Larger and UNIFORM font sizes (including confusion matrix numbers, x/y labels,
   and waveform/probability example plots):
      --font_base (default 18)

4) Feature importance plot:
      only Top-K features (default K=5 via --topk)

5) Editable formats:
      For EVERY plot we save:
        - PNG (high dpi)
        - PDF (vector, editable)
        - SVG (vector, editable)

6) Two copies for EVERY plot:
      - normal copy with a title at the top
      - a second copy with NO title at the top (suffix: _notitle)

7) Waveform + probability "example_detection" plots are saved (titled + notitle)
   into:
      <out_dir>/<plots_subdir>/examples/

8) A SINGLE combined figure (2x2 grid) from 4 summary plots:
      - segment confusion matrix
      - feature importances (topK)
      - sentence confusion matrix
      - sentence max-prob histogram

   We also save TWO combined panels:
      - <out_dir>/<combine_name>.{png,pdf,svg}          (titled)
      - <out_dir>/<combine_name>_notitle.{png,pdf,svg}  (no title)

NOTE (your request):
--------------------
All explicit plotting colors are constrained to ONLY:
    RED, GREEN, BLUE
(no inferno/orange/other theme colors).

Default expected ZIPs (inside --input_dir):
- Red_story_231125.zip
- not_RED_speaking.zip
- 5 times Red.zip
- Red 100x 10Hz.zip
- Red 100x 13Hz.zip

Run examples:
-------------
Save 12 waveform examples (default):
  python red_detect_rf.py --input_dir ./zips --out_dir ./out

Save ALL waveform examples from the sentence-level test set:
  python red_detect_rf.py --input_dir ./zips --out_dir ./out --n_example_plots -1
"""

import os
import zipfile
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Matplotlib (non-interactive backend)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ----------------------------------------------------------------------
# Color policy: ONLY RED / GREEN / BLUE for plotted elements
# ----------------------------------------------------------------------

C_RED = "red"
C_GREEN = "green"
C_BLUE = "blue"

# Discrete 3-color map for confusion matrices (0–100%)
CMAP_RGB3 = ListedColormap([C_BLUE, C_GREEN, C_RED])
NORM_0_100_RGB3 = BoundaryNorm([0.0, 33.333, 66.666, 100.0001], CMAP_RGB3.N)


def _cm_text_color_from_percent(pct: float) -> str:
    """
    Pick a text color (ONLY RGB) that contrasts with the discrete CM background.
    Background bins:
      [0,33.33)   -> BLUE    => text RED
      [33.33,66.66) -> GREEN => text BLUE
      [66.66,100] -> RED     => text GREEN
    """
    if pct < 33.333:
        return C_RED
    elif pct < 66.666:
        return C_BLUE
    else:
        return C_GREEN


# ----------------------------------------------------------------------
# Helpers: paths, dirs, and saving figures
# ----------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def notitle_path(p: str) -> str:
    base, ext = os.path.splitext(p)
    return base + "_notitle" + ext


def save_figure_multi(
    fig: plt.Figure,
    out_png: str,
    dpi: int,
    also_pdf: bool = True,
    also_svg: bool = True
) -> None:
    """
    Save a matplotlib figure as:
      - PNG (high DPI)
      - PDF (editable vector)
      - SVG (editable vector)
    """
    out_png = os.path.abspath(out_png)
    base, _ = os.path.splitext(out_png)
    ensure_dir(os.path.dirname(out_png))

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    if also_pdf:
        fig.savefig(base + ".pdf", bbox_inches="tight")
    if also_svg:
        fig.savefig(base + ".svg", bbox_inches="tight")


# ----------------------------------------------------------------------
# Plot style (uniform fonts everywhere)
# ----------------------------------------------------------------------

def set_plot_style(font_base: int) -> Dict[str, int]:
    """
    Define a consistent font system so all plots look uniform.
    Returns a dict of font sizes used everywhere.
    """
    fs = {
        "BASE": font_base,
        "TITLE": font_base + 4,
        "LABEL": font_base + 2,
        "TICK": font_base,
        "ANNOT": font_base + 1,
        "LEGEND": font_base,
        "CBAR_LABEL": font_base + 1,
        "CBAR_TICK": font_base,
    }

    plt.rcParams.update({
        "font.size": fs["BASE"],
        "axes.titlesize": fs["TITLE"],
        "axes.labelsize": fs["LABEL"],
        "xtick.labelsize": fs["TICK"],
        "ytick.labelsize": fs["TICK"],
        "legend.fontsize": fs["LEGEND"],
        "figure.titlesize": fs["TITLE"],
    })
    return fs


# ----------------------------------------------------------------------
# Basic IO
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Load sentence-level dataset (RED / non-RED stories)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Pulse detection utilities (single-word RED recordings)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Feature extraction + feature names
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Template-based helper for RED segments inside sentences
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Single-word RED file discovery (ZIP-based)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Plot helpers (save both titled + notitle variants)
# ----------------------------------------------------------------------

def plot_confusion_matrix_percent_dual(
    cm: np.ndarray,
    class_names: List[str],
    out_png: str,
    title: str,
    dpi: int,
    fs: Dict[str, int]
) -> None:
    """
    Save TWO versions:
      - out_png (titled)
      - out_png with '_notitle' suffix (no title)
    Each saved as PNG+PDF+SVG.

    Color policy: ONLY RGB (3-level discrete colormap).
    """
    def _plot(show_title: bool) -> plt.Figure:
        cm_f = cm.astype(np.float32)
        row_sums = cm_f.sum(axis=1, keepdims=True)
        cm_perc = np.zeros_like(cm_f, dtype=np.float32)
        np.divide(cm_f, row_sums, out=cm_perc, where=row_sums != 0)
        cm_perc *= 100.0

        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        im = ax.imshow(cm_perc, interpolation="nearest", cmap=CMAP_RGB3, norm=NORM_0_100_RGB3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percentage (%)", fontsize=fs["CBAR_LABEL"])
        cbar.ax.tick_params(labelsize=fs["CBAR_TICK"])

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.set_yticklabels(class_names)
        ax.tick_params(axis="both", labelsize=fs["TICK"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = float(cm_perc[i, j])
                text = f"{pct:.1f}%\n({int(cm[i, j])})"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color=_cm_text_color_from_percent(pct),
                    fontsize=fs["ANNOT"]
                )

        ax.set_xlabel("Predicted label", fontsize=fs["LABEL"])
        ax.set_ylabel("True label", fontsize=fs["LABEL"])
        if show_title:
            ax.set_title(title, fontsize=fs["TITLE"])

        fig.tight_layout()
        return fig

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_png, dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, notitle_path(out_png), dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


def plot_feature_importances_dual(
    rf: RandomForestClassifier,
    feature_names: List[str],
    out_png: str,
    dpi: int,
    fs: Dict[str, int],
    top_k: int = 5
) -> None:
    """
    Save TWO versions (titled + notitle) of Top-K feature importances.

    Color policy: ONLY RGB (bars in BLUE).
    """
    importances = rf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top_idx = idx_sorted[:top_k]
    top_importances = importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    def _plot(show_title: bool) -> plt.Figure:
        fig_h = max(3.5, 0.75 * top_k + 2.0)
        fig, ax = plt.subplots(figsize=(9.0, fig_h))

        y_pos = np.arange(len(top_idx))
        ax.barh(y_pos, top_importances, color=C_BLUE, edgecolor=C_BLUE)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=fs["TICK"])
        ax.invert_yaxis()

        ax.set_xlabel("Importance", fontsize=fs["LABEL"])
        ax.tick_params(axis="x", labelsize=fs["TICK"])

        if show_title:
            ax.set_title(f"Top {top_k} RF feature importances", fontsize=fs["TITLE"])

        fig.tight_layout()
        return fig

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_png, dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, notitle_path(out_png), dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


def plot_sentence_maxprob_hist_dual(
    max_red: List[float],
    max_notred: List[float],
    out_png: str,
    dpi: int,
    fs: Dict[str, int],
    title: str = "Separability of RED vs non-RED sentences"
) -> None:
    """
    Save TWO versions (titled + notitle) of histogram.

    Color policy: ONLY RGB (RED vs BLUE).
    """
    def _plot(show_title: bool) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        if len(max_red) > 0:
            ax.hist(max_red, bins=10, alpha=0.55, density=True, color=C_RED, label="RED sentences")
        if len(max_notred) > 0:
            ax.hist(max_notred, bins=10, alpha=0.55, density=True, color=C_BLUE, label="Non-RED sentences")

        ax.set_xlabel("Max p(RED) over sentence", fontsize=fs["LABEL"])
        ax.set_ylabel("Density", fontsize=fs["LABEL"])
        ax.tick_params(axis="both", labelsize=fs["TICK"])
        ax.legend(loc="best", fontsize=fs["LEGEND"])

        if show_title:
            ax.set_title(title, fontsize=fs["TITLE"])

        fig.tight_layout()
        return fig

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_png, dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, notitle_path(out_png), dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


def plot_pulse_duration_hist_dual(
    all_durations: np.ndarray,
    mean_dur: float,
    out_png: str,
    dpi: int,
    fs: Dict[str, int],
    title: str = "Distribution of RED pulse durations (single-word data)"
) -> None:
    """
    Save TWO versions (titled + notitle) of pulse duration histogram.

    Color policy: ONLY RGB (bars BLUE, mean line RED).
    """
    def _plot(show_title: bool) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        ax.hist(all_durations, bins=30, alpha=0.75, color=C_BLUE)
        ax.axvline(mean_dur, linestyle="--", linewidth=2, color=C_RED, label=f"Mean = {mean_dur:.2f} s")

        ax.set_xlabel("Pulse duration (s)", fontsize=fs["LABEL"])
        ax.set_ylabel("Count", fontsize=fs["LABEL"])
        ax.tick_params(axis="both", labelsize=fs["TICK"])
        ax.legend(fontsize=fs["LEGEND"])

        if show_title:
            ax.set_title(title, fontsize=fs["TITLE"])

        fig.tight_layout()
        return fig

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_png, dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, notitle_path(out_png), dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


def plot_detection_example_dual(
    signal: np.ndarray,
    sr: float,
    label: int,
    pred_has_red: int,
    meta: Dict[str, Any],
    centers_t: np.ndarray,
    probs: np.ndarray,
    clusters: List[Dict[str, Any]],
    out_png: str,
    dpi: int,
    fs: Dict[str, int],
    seg_threshold: float
) -> None:
    """
    Save TWO versions (titled + notitle) of the waveform + probability plot.

    Color policy: ONLY RGB
      - waveform: BLUE
      - event spans: GREEN (alpha)
      - prob curve: BLUE
      - threshold line: RED
    """
    t = np.arange(len(signal)) / sr
    x = normalize_signal(signal)

    def _plot(show_title: bool) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]}
        )

        # Waveform
        ax1.plot(t, x, linewidth=1.0, color=C_BLUE)
        for cl in clusters:
            ax1.axvspan(cl["start_s"], cl["end_s"], color=C_GREEN, alpha=0.20)
        ax1.set_ylabel("Amplitude (norm.)", fontsize=fs["LABEL"])
        ax1.tick_params(axis="both", labelsize=fs["TICK"])

        if show_title:
            sent_name = meta.get("sentence", "UNKNOWN")
            ax1.set_title(
                f"Waveform | Sentence: {sent_name} | True RED={label} | Pred has_RED={pred_has_red}",
                fontsize=fs["TITLE"]
            )

        # Probability curve
        ax2.plot(centers_t, probs, linewidth=2.0, color=C_BLUE)
        ax2.axhline(
            seg_threshold,
            color=C_RED,
            linestyle="--",
            linewidth=2.5,
            label=f"Threshold={seg_threshold:.2f}"
        )

        for cl in clusters:
            ax2.axvspan(cl["start_s"], cl["end_s"], color=C_GREEN, alpha=0.20)
            mid = 0.5 * (cl["start_s"] + cl["end_s"])
            ax2.text(
                mid,
                min(1.0, cl["mean_prob"] + 0.05),
                f"{cl['mean_prob']:.2f}",
                ha="center",
                va="bottom",
                fontsize=fs["ANNOT"],
                color=C_BLUE
            )

        ax2.set_xlabel("Time (s)", fontsize=fs["LABEL"])
        ax2.set_ylabel("p(RED | window)", fontsize=fs["LABEL"])
        ax2.set_ylim(0.0, 1.02)
        ax2.tick_params(axis="both", labelsize=fs["TICK"])
        ax2.legend(loc="upper right", fontsize=fs["LEGEND"])

        fig.tight_layout()
        return fig

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_png, dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, notitle_path(out_png), dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


# ----------------------------------------------------------------------
# Training set construction from single-word RED zips
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Sliding window + clustering over sentences
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


# ----------------------------------------------------------------------
# Combined 2x2 panel (titled + notitle)
# ----------------------------------------------------------------------

def combine_four_images_2x2_panel_dual(
    img_paths: List[str],
    out_base: str,
    dpi: int,
    fs: Dict[str, int],
    title_text: str = "RED RF summary (2×2)"
) -> None:
    """
    Combine 4 existing PNG images into one 2x2 panel using matplotlib.
    Saves TWO versions:
      - out_base.{png,pdf,svg}          (titled)
      - out_base_notitle.{png,pdf,svg}  (no title)
    """
    if len(img_paths) != 4:
        raise ValueError("img_paths must have exactly 4 items.")
    for p in img_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image for combine: {p}")

    imgs = [plt.imread(p) for p in img_paths]

    def _plot(show_title: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        for ax, im in zip(axes, imgs):
            ax.imshow(im)
            ax.axis("off")

        if show_title:
            fig.suptitle(title_text, fontsize=fs["TITLE"])
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()
        return fig

    out_base = os.path.abspath(out_base)
    ensure_dir(os.path.dirname(out_base))

    fig1 = _plot(show_title=True)
    save_figure_multi(fig1, out_base + ".png", dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig1)

    fig2 = _plot(show_title=False)
    save_figure_multi(fig2, out_base + "_notitle.png", dpi=dpi, also_pdf=True, also_svg=True)
    plt.close(fig2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=".",
                    help="Folder containing all required ZIP files.")
    ap.add_argument("--out_dir", type=str, default=".",
                    help="Where to write outputs.")
    ap.add_argument("--plots_subdir", type=str, default="plots_red_rf_cluster",
                    help="Subfolder under --out_dir for plots.")
    ap.add_argument("--combine_name", type=str, default="red_rf_summary_2x2",
                    help="Base name (no extension) for combined 2x2 panel.")
    ap.add_argument("--dpi", type=int, default=600,
                    help="DPI for PNG outputs (PDF/SVG are vector).")
    ap.add_argument("--font_base", type=int, default=18,
                    help="Base font size used across all figures.")
    ap.add_argument("--topk", type=int, default=5,
                    help="Top-K features to show in feature importance plot.")
    ap.add_argument("--n_example_plots", type=int, default=12,
                    help="How many waveform example plots to save from test set. Use -1 for ALL.")
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.out_dir)
    plots_dir = os.path.join(out_dir, args.plots_subdir)
    ensure_dir(plots_dir)

    fs = set_plot_style(args.font_base)

    examples_dir = os.path.join(plots_dir, "examples")
    ensure_dir(examples_dir)

    red_zip = os.path.join(input_dir, "Red_story_231125.zip")
    not_red_zip = os.path.join(input_dir, "not_RED_speaking.zip")

    if not os.path.exists(red_zip):
        raise FileNotFoundError(f"Missing: {red_zip}")
    if not os.path.exists(not_red_zip):
        raise FileNotFoundError(f"Missing: {not_red_zip}")

    print("========================================")
    print("RED RF pipeline")
    print(f"Input dir        : {input_dir}")
    print(f"Output dir       : {out_dir}")
    print(f"Plots dir        : {plots_dir}")
    print(f"Examples dir     : {examples_dir}")
    print(f"DPI              : {args.dpi}")
    print(f"Font base        : {args.font_base}")
    print(f"Top-K FI         : {args.topk}")
    print(f"n_example_plots  : {args.n_example_plots}")
    print("========================================")

    print("\nLoading sentence-level dataset...")
    signals, labels, metas, sr = load_sentences(red_zip, not_red_zip)
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)

    print(f"Loaded {len(signals)} sentences (sr ≈ {sr:.1f} Hz, length ≈ {sig_len/sr:.2f} s).")
    print(f"RED sentences: {int(labels.sum())}, non-RED: {len(labels) - int(labels.sum())}")

    single_members = find_single_red_files(input_dir)
    if not single_members:
        raise RuntimeError(
            "No single-word RED CSV files found.\n"
            "Expected zip archives like '5 times Red.zip', 'Red 100x 10Hz.zip', "
            "'Red 100x 13Hz.zip' in --input_dir, each containing CSV recordings."
        )

    print("\nDetected single-word RED zip members (showing up to 20):")
    for zp, mem in single_members[:20]:
        print(f"  {os.path.basename(zp)} :: {mem}")
    if len(single_members) > 20:
        print(f"  ... ({len(single_members)-20} more)")

    red_segments_single, mean_pulse_dur, min_pulse_dur, max_pulse_dur, seg_len = build_red_segments_from_single_files(
        single_members,
        sr_assumed=sr,
        plots_dir=plots_dir,
        dpi=args.dpi,
        fs=fs
    )
    print(f"Segment length used for RF: {seg_len} samples (~{seg_len/sr:.3f} s).")

    scaler, rf = build_training_sets(
        red_segments_single=red_segments_single,
        sentence_signals=signals,
        sentence_labels=labels,
        sr=sr,
        seg_len=seg_len,
        plots_dir=plots_dir,
        dpi=args.dpi,
        fs=fs,
        top_k_features=args.topk
    )

    idx_all = np.arange(len(signals))
    _, idx_test_sent = train_test_split(
        idx_all, test_size=0.3, stratify=labels, random_state=42
    )

    seg_threshold = 0.6
    y_true_sent = labels[idx_test_sent]
    y_pred_sent = []

    max_probs_red: List[float] = []
    max_probs_notred: List[float] = []

    print("\n=== Sentence-level RED detection on test set ===")
    for i, idx in enumerate(idx_test_sent):
        sig = signals[idx]
        lab = int(labels[idx])
        meta = metas[idx]

        centers_t, probs = sliding_window_probs_rf(sig, sr, seg_len, rf, scaler, stride_sec=0.01)
        clusters = cluster_high_prob_windows(
            centers_t, probs, seg_len, sr,
            seg_threshold=seg_threshold,
            min_pulse_dur=min_pulse_dur,
            max_pulse_dur=max_pulse_dur,
            max_gap_sec=0.03
        )

        has_red_pred = int(len(clusters) > 0)
        y_pred_sent.append(has_red_pred)

        if len(probs) > 0:
            max_p = float(np.max(probs))
            if lab == 1:
                max_probs_red.append(max_p)
            else:
                max_probs_notred.append(max_p)

        times_str = ", ".join(
            [f"[{cl['start_s']:.2f}, {cl['end_s']:.2f}] (p≈{cl['mean_prob']:.2f})"
             for cl in clusters]
        ) if clusters else "None"
        print(f"Sentence {i+1:02d} | True RED={lab} | Pred has_RED={has_red_pred} "
              f"| Pred count={len(clusters)} | Events: {times_str}")

        n_to_plot = args.n_example_plots
        do_plot = (n_to_plot == -1) or (i < n_to_plot)
        if do_plot:
            out_png = os.path.join(
                examples_dir,
                f"example_detection_{i+1:02d}_TRUE{lab}_PRED{has_red_pred}.png"
            )
            plot_detection_example_dual(
                signal=sig,
                sr=sr,
                label=lab,
                pred_has_red=has_red_pred,
                meta=meta,
                centers_t=centers_t,
                probs=probs,
                clusters=clusters,
                out_png=out_png,
                dpi=args.dpi,
                fs=fs,
                seg_threshold=seg_threshold
            )

    y_true_sent = np.array(y_true_sent, dtype=int)
    y_pred_sent = np.array(y_pred_sent, dtype=int)

    cm_sent = confusion_matrix(y_true_sent, y_pred_sent)
    print("\nSentence-level confusion matrix (rows=true, cols=pred):")
    print(cm_sent)
    print("\nSentence-level classification report:")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))

    plot_confusion_matrix_percent_dual(
        cm_sent,
        class_names=["Non-RED sentence", "RED sentence"],
        out_png=os.path.join(plots_dir, "sentence_confusion_matrix_percent.png"),
        title="Sentence-level RF confusion matrix",
        dpi=args.dpi,
        fs=fs
    )

    plot_sentence_maxprob_hist_dual(
        max_probs_red,
        max_probs_notred,
        out_png=os.path.join(plots_dir, "sentence_maxprob_hist_red_vs_notred.png"),
        dpi=args.dpi,
        fs=fs
    )

    img1 = os.path.join(plots_dir, "segment_rf_confusion_matrix_percent.png")
    img2 = os.path.join(plots_dir, "segment_rf_feature_importances.png")
    img3 = os.path.join(plots_dir, "sentence_confusion_matrix_percent.png")
    img4 = os.path.join(plots_dir, "sentence_maxprob_hist_red_vs_notred.png")

    combined_base = os.path.join(out_dir, args.combine_name)
    combine_four_images_2x2_panel_dual([img1, img2, img3, img4], combined_base, dpi=args.dpi, fs=fs)

    print("\n========================================")
    print(f"Plots saved into: {plots_dir}")
    print(f"Waveform examples: {examples_dir}")
    print(f"Combined 2x2 saved: {combined_base}.png/.pdf/.svg and {combined_base}_notitle.png/.pdf/.svg")
    print("Done.")
    print("========================================")


if __name__ == "__main__":
    main()
