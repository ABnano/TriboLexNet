#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_word_detection.py

RED word *detection* (time localization) in piezoelectric acoustic recordings.

Pipeline:
- Load recordings from:
    * Red_story_231125.zip  (sentences containing the word "RED")
    * not_RED_speaking.zip  (sentences without "RED")
- Stage 1: Build an initial template of the word "RED" from
           "RED before story" and "RED after the story" sentences
           using a simple energy-based heuristic.
- Stage 2: Refine the template using normalized cross-correlation:
    * For every RED sentence, detect the RED position via NCC peak.
    * Extract a fixed-length segment around that peak.
    * Average aligned segments to obtain a refined template.
- Stage 3: Analyze stability and separability:
    * Overlay all aligned RED segments (waveform).
    * Compute and plot the average spectrogram of RED segments.
    * Compute the max NCC score for RED vs NOT-RED sentences and
      plot their histograms.
- Plots are saved in a "plots_red_word" directory.

The script also provides:
    detect_red_positions(signal, sr, template, threshold)
for detecting RED locations in a new sentence.
"""

import os
import zipfile
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Use non-interactive backend so Tk/Tkinter is never touched
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# Basic IO & signal utilities
# ---------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current) arrays.

    The files use:
        - ';' as separator
        - ',' as decimal
        - two header lines + one blank line, then data
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

    Returns:
        signals: list of 1D float32 arrays
        labels:  list of ints (1 for RED, 0 for NOT-RED)
        metas:   list of dicts with metadata (sentence/category, path, etc.)
        sr:      sampling rate (Hz) estimated from the first RED file
    """
    signals: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[Dict] = []

    def _load_one(zip_path: str, label: int) -> None:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = sorted([m for m in z.namelist() if m.lower().endswith('.csv')])
            for m in members:
                # For red zip: "Red story (1)/Sentence text/file.csv"
                # For not-red zip: "sentence text/file.csv"
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

    # Estimate sampling rate from first RED recording
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


# ---------------------------------------------------------------
# Normalized cross-correlation & segment extraction
# ---------------------------------------------------------------

def normalized_cross_correlation(template: np.ndarray,
                                 signal: np.ndarray) -> np.ndarray:
    """
    Compute normalized cross-correlation between template and signal.

    Both inputs are 1D arrays. The template is zero-meaned internally.
    Returns:
        ncc: 1D array of length len(signal) - len(template) + 1
             values roughly in [-1, 1].
    """
    t = np.asarray(template, dtype=np.float32)
    x = np.asarray(signal, dtype=np.float32)

    L = len(t)
    if L < 1 or len(x) < L:
        raise ValueError("Template must be shorter than signal.")

    # Zero-mean template
    t = t - np.mean(t)
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-8:
        raise ValueError("Template norm is too small.")

    # Unnormalized correlation via FFT-based correlate
    c = np.correlate(x, t, mode='valid')  # length: N - L + 1

    # Sliding window statistics of x
    x2 = x ** 2
    cumsum_x = np.concatenate(([0.0], np.cumsum(x)))
    cumsum_x2 = np.concatenate(([0.0], np.cumsum(x2)))
    N = len(x)
    out_len = N - L + 1

    sum_x = cumsum_x[L:] - cumsum_x[:-L]     # length out_len
    sum_x2 = cumsum_x2[L:] - cumsum_x2[:-L]  # length out_len

    # sum((x - mean)^2) = sum(x^2) - sum(x)^2 / L
    var_seg = sum_x2 - (sum_x ** 2) / float(L)
    var_seg = np.maximum(var_seg, 1e-8)
    std_seg = np.sqrt(var_seg)

    # NCC
    ncc = c / (norm_t * std_seg)
    return ncc.astype(np.float32)


def extract_segment_around_center(signal: np.ndarray,
                                  center_idx: int,
                                  seg_length: int) -> np.ndarray:
    """
    Extract a segment of length seg_length around a given center index.
    If needed, pad with zeros at the edges.
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


def detect_red_positions(signal: np.ndarray,
                         sr: float,
                         template: np.ndarray,
                         threshold: float = 0.4) -> List[Tuple[float, float]]:
    """
    Detect positions of the word "RED" in a given signal using normalized
    cross-correlation with the learned template.

    Args:
        signal: 1D waveform
        sr: sampling rate (Hz)
        template: learned RED template (1D array)
        threshold: NCC threshold for detection (0..1)

    Returns:
        List of (t_start, t_end) in seconds for each detected RED segment.
        For this dataset there is effectively at most one RED, but the
        function is written to return a list for generality.
    """
    ncc = normalized_cross_correlation(template, signal)
    max_idx = int(np.argmax(ncc))
    max_val = float(ncc[max_idx])
    if max_val < threshold:
        # No confident detection
        return []

    L = len(template)
    center = max_idx + L // 2
    t_center = center / sr
    duration = L / sr
    t_start = t_center - duration / 2.0
    t_end = t_center + duration / 2.0
    return [(t_start, t_end)]


# ---------------------------------------------------------------
# Template construction & refinement
# ---------------------------------------------------------------

# These are the folder names for the simple RED sentences
RED_TEMPLATE_SENTENCES = ("RED before story", "RED after the story")


def build_initial_template(signals: List[np.ndarray],
                           metas: List[Dict],
                           sr: float,
                           window_duration: float = 0.6) -> np.ndarray:
    """
    Build an initial template for the word "RED" using the sentences
    "RED before story" and "RED after the story".

    Uses a simple energy-based heuristic to locate the main energy
    region and extracts a fixed-duration window around it, then averages.
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
        raise RuntimeError(
            "No segments found for building initial RED template. "
            "Check that RED_TEMPLATE_SENTENCES match your folder names."
        )

    template = np.mean(np.stack(segments, axis=0), axis=0)
    template = template - np.mean(template)
    norm = np.linalg.norm(template)
    if norm > 1e-8:
        template = template / norm
    return template.astype(np.float32)


def refine_template_with_all_red(signals: List[np.ndarray],
                                 labels: np.ndarray,
                                 sr: float,
                                 initial_template: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Use normalized cross-correlation with the initial template to find
    RED positions in all RED sentences, extract aligned segments, and
    build a refined template as their mean.

    Returns:
        refined_template
        segments: list of aligned RED segments (waveforms)
    """
    red_indices = np.where(labels == 1)[0]
    seg_length = len(initial_template)
    segments = []

    for idx in red_indices:
        x = normalize_signal(signals[idx])
        ncc = normalized_cross_correlation(initial_template, x)
        peak_idx = int(np.argmax(ncc))
        center = peak_idx + seg_length // 2
        seg = extract_segment_around_center(x, center, seg_length)
        segments.append(seg)

    if not segments:
        raise RuntimeError("No RED segments found during refinement.")

    segments_arr = np.stack(segments, axis=0)
    refined = np.mean(segments_arr, axis=0)
    refined = refined - np.mean(refined)
    norm = np.linalg.norm(refined)
    if norm > 1e-8:
        refined = refined / norm
    return refined.astype(np.float32), segments


# ---------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------

def plot_example_red_sentences_with_energy(signals: List[np.ndarray],
                                           metas: List[Dict],
                                           sr: float,
                                           plots_dir: str,
                                           max_examples: int = 4) -> None:
    """
    Plot a few example RED sentences with energy envelope and mark
    the maximum-energy point (heuristic RED location).
    """
    ensure_dir(plots_dir)
    red_examples = []
    for x, m in zip(signals, metas):
        if m.get('sentence', '') in RED_TEMPLATE_SENTENCES:
            red_examples.append((x, m))
        if len(red_examples) >= max_examples:
            break

    if not red_examples:
        return

    fig, axes = plt.subplots(len(red_examples), 1, figsize=(8, 2.5 * len(red_examples)), sharex=True)
    if len(red_examples) == 1:
        axes = [axes]

    for ax, (s, m) in zip(axes, red_examples):
        x = normalize_signal(s)
        t = np.arange(len(x)) / sr
        abs_x = np.abs(x)
        kernel_size = max(1, int(0.03 * sr))
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        env = np.convolve(abs_x, kernel, mode='same')
        centre = int(np.argmax(env))
        t_c = centre / sr

        ax.plot(t, x, label="Waveform")
        ax.plot(t, env / np.max(np.abs(env) + 1e-8), label="Energy envelope (normalized)", alpha=0.8)
        ax.axvline(t_c, color="red", linestyle="--", label="Max energy ~ RED")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(f"Example RED sentence: {m.get('sentence', 'UNKNOWN')}")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out = os.path.join(plots_dir, "stage1_example_red_sentences_energy.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_template_waveform(template: np.ndarray,
                           sr: float,
                           plots_dir: str,
                           name: str) -> None:
    """Plot the template waveform."""
    ensure_dir(plots_dir)
    t = np.arange(len(template)) / sr
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, template)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"{name} waveform")
    fig.tight_layout()
    out = os.path.join(plots_dir, f"{name.replace(' ', '_').lower()}_waveform.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_template_spectrogram(template: np.ndarray,
                              sr: float,
                              plots_dir: str,
                              name: str) -> None:
    """Plot log-power spectrogram of the template."""
    ensure_dir(plots_dir)
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
    ax.set_title(f"{name} log-power spectrogram")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(plots_dir, f"{name.replace(' ', '_').lower()}_spectrogram.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_overlay_red_segments(segments: List[np.ndarray],
                              sr: float,
                              plots_dir: str) -> None:
    """
    Plot all RED segments overlaid, plus mean ± std envelope,
    to visually demonstrate stability.
    """
    ensure_dir(plots_dir)
    if not segments:
        return
    seg_arr = np.stack(segments, axis=0)
    mean_seg = np.mean(seg_arr, axis=0)
    std_seg = np.std(seg_arr, axis=0)

    t = np.arange(seg_arr.shape[1]) / sr
    fig, ax = plt.subplots(figsize=(7, 4))

    # Overlay all segments (light)
    for s in seg_arr:
        ax.plot(t, s, color="gray", alpha=0.3, linewidth=0.7)

    # Mean ± std
    ax.plot(t, mean_seg, color="blue", linewidth=2.0, label="Mean RED segment")
    ax.fill_between(t, mean_seg - std_seg, mean_seg + std_seg,
                    color="blue", alpha=0.2, label="Mean ± 1 std")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title("Aligned RED segments across sentences")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(plots_dir, "stage2_overlay_red_segments.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_average_red_spectrogram(segments: List[np.ndarray],
                                 sr: float,
                                 plots_dir: str) -> None:
    """
    Compute spectrogram for each RED segment, then plot the average
    spectrogram across all segments.
    """
    ensure_dir(plots_dir)
    if not segments:
        return

    specs = []
    min_frames = None
    for s in segments:
        spec = compute_log_spectrogram(s, n_fft=256, hop_length=128)
        if min_frames is None or spec.shape[1] < min_frames:
            min_frames = spec.shape[1]
        specs.append(spec)
    # Crop all to same number of frames
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
    ax.set_title("Average log-power spectrogram of RED segments")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(plots_dir, "stage2_average_red_spectrogram.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_corr_hist_red_vs_notred(signals: List[np.ndarray],
                                 labels: np.ndarray,
                                 template: np.ndarray,
                                 plots_dir: str) -> None:
    """
    Compute max NCC for each signal using the (refined) template,
    and plot histograms for RED vs NOT-RED.
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_idx = np.where(labels == 1)[0]
    notred_idx = np.where(labels == 0)[0]

    max_red = []
    max_notred = []

    for i in red_idx:
        x = normalize_signal(signals[i])
        ncc = normalized_cross_correlation(template, x)
        max_red.append(float(np.max(ncc)))

    for i in notred_idx:
        x = normalize_signal(signals[i])
        ncc = normalized_cross_correlation(template, x)
        max_notred.append(float(np.max(ncc)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(max_red, bins=10, alpha=0.7, label="RED sentences", density=True)
    ax.hist(max_notred, bins=10, alpha=0.7, label="Non-RED sentences", density=True)
    ax.set_xlabel("Max normalized cross-correlation with RED template")
    ax.set_ylabel("Density")
    ax.set_title("Separability of RED vs non-RED by template-matching")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(plots_dir, "stage3_ncc_hist_red_vs_notred.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_example_detection(signals: List[np.ndarray],
                           labels: np.ndarray,
                           metas: List[Dict],
                           sr: float,
                           template: np.ndarray,
                           plots_dir: str,
                           max_examples_each: int = 2) -> None:
    """
    Plot a few RED and non-RED sentences with normalized correlation
    curve and detected RED region (if above threshold).
    """
    ensure_dir(plots_dir)
    labels = np.asarray(labels, dtype=int)
    red_idx = np.where(labels == 1)[0][:max_examples_each]
    notred_idx = np.where(labels == 0)[0][:max_examples_each]

    indices = list(red_idx) + list(notred_idx)
    if not indices:
        return

    threshold = 0.4  # heuristic, can be tuned

    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 2.5 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        x = normalize_signal(signals[idx])
        m = metas[idx]
        sent = m.get('sentence', 'UNKNOWN')
        lab = labels[idx]
        t = np.arange(len(x)) / sr

        # Waveform
        ax_w = axes[row, 0]
        ax_w.plot(t, x, linewidth=0.8)
        ax_w.set_ylabel("Amplitude")
        ax_w.set_title(f"Waveform ({'RED' if lab == 1 else 'non-RED'}): {sent}")
        ax_w.set_xlim([t[0], t[-1]])

        # NCC curve
        ncc = normalized_cross_correlation(template, x)
        tt = np.arange(len(ncc)) / sr
        ax_c = axes[row, 1]
        ax_c.plot(tt, ncc, linewidth=0.8, label="NCC")
        ax_c.axhline(threshold, color="red", linestyle="--", label="Threshold")
        max_idx = int(np.argmax(ncc))
        max_val = float(ncc[max_idx])
        t_peak = max_idx / sr
        ax_c.axvline(t_peak, color="green", linestyle="--", label="Peak NCC")

        ax_c.set_ylabel("NCC")
        ax_c.set_xlabel("Time (s)")
        ax_c.set_title(f"NCC vs time (max={max_val:.3f})")
        ax_c.set_xlim([tt[0], tt[-1]])
        ax_c.legend(fontsize=7, loc="upper right")

    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    out = os.path.join(plots_dir, "stage3_example_detection_waveform_and_ncc.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def run_red_word_detection(red_zip: str,
                           not_red_zip: str,
                           plots_dir: str = "plots_red_word") -> None:
    ensure_dir(plots_dir)

    print("Loading dataset...")
    signals, labels, metas, sr = load_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ~ {sr:.1f} Hz).")

    # Normalize and crop all signals to common length
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped all signals to common length: {sig_len} samples (~{sig_len/sr:.2f} s).")

    labels = np.asarray(labels, dtype=int)

    # Stage 1: initial template & example energy plots
    print("Stage 1: Building initial RED template from simple sentences...")
    plot_example_red_sentences_with_energy(signals, metas, sr, plots_dir)
    initial_template = build_initial_template(signals, metas, sr, window_duration=0.6)
    plot_template_waveform(initial_template, sr, plots_dir, name="Initial_RED_template")
    plot_template_spectrogram(initial_template, sr, plots_dir, name="Initial_RED_template")

    # Stage 2: refine template with all RED sentences
    print("Stage 2: Refining template using all RED sentences via NCC alignment...")
    refined_template, red_segments = refine_template_with_all_red(signals, labels, sr, initial_template)
    plot_template_waveform(refined_template, sr, plots_dir, name="Refined_RED_template")
    plot_template_spectrogram(refined_template, sr, plots_dir, name="Refined_RED_template")
    plot_overlay_red_segments(red_segments, sr, plots_dir)
    plot_average_red_spectrogram(red_segments, sr, plots_dir)

    # Stage 3: Detection behavior & separability
    print("Stage 3: Analyzing detection behavior and separability...")
    plot_corr_hist_red_vs_notred(signals, labels, refined_template, plots_dir)
    plot_example_detection(signals, labels, metas, sr, refined_template, plots_dir)

    print(f"All plots saved in: {os.path.abspath(plots_dir)}")
    print("You can now use 'detect_red_positions' with the refined template "
          "for new/unknown sentences.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RED word detection and analysis in piezoelectric acoustic signals."
    )
    parser.add_argument('--red_zip', type=str, default='Red_story_231125.zip',
                        help='Path to zip with RED stories.')
    parser.add_argument('--not_red_zip', type=str, default='not_RED_speaking.zip',
                        help='Path to zip with non-RED stories.')
    parser.add_argument('--plots_dir', type=str, default='plots_red_word',
                        help='Directory to save plots.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_red_word_detection(
        red_zip=args.red_zip,
        not_red_zip=args.not_red_zip,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
