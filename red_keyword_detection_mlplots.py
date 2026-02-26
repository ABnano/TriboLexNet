#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_keyword_detection_mlplots.py

Fast ML (no neural nets) training curves for detecting the word "RED" in
piezoelectric acoustic recordings.

Key idea:
- Use SGDClassifier(loss="log_loss") so we can train epoch-by-epoch with partial_fit.
- Record train/val loss + AUC for 100 epochs.
- Sweep learning rates to generate “nice training plots” for manuscripts.

Outputs:
- out_dir/*.png (400 dpi, big fonts)
- out_dir/*.pdf (vector-friendly)
- out_dir/*.csv (metrics log per run)

Usage example:
  python red_keyword_detection_mlplots.py \
    --red_zip data/Red_story_231125.zip \
    --not_red_zip data/not_RED_speaking.zip \
    --feature_set combined \
    --epochs 100 \
    --lrs 0.001,0.003,0.01,0.03 \
    --out_dir outputs/results_mlplots
"""

import os
import argparse
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss

import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# I/O + preprocessing
# ---------------------------------------------------------------

def read_signal_from_zip(zip_path: str, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file from the given zip archive and return (time, current) arrays.

    Format assumptions:
      - ';' separator
      - ',' decimal
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
        sr:      sampling rate (Hz) estimated from the first file
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

    # Estimate sampling rate from first recording
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


# ---------------------------------------------------------------
# Approach 1 feature: template correlation
# ---------------------------------------------------------------

RED_TEMPLATE_SENTENCES = ("RED before story", "RED after the story")


def build_red_template(signals: List[np.ndarray],
                       metas: List[Dict],
                       sr: float,
                       window_duration: float = 0.6) -> np.ndarray:
    """
    Build a template for the word "RED" using two known sentence folders.
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
            "No segments found for building RED template. "
            "Check that RED_TEMPLATE_SENTENCES match your folder names."
        )

    template = np.mean(np.stack(segments, axis=0), axis=0)
    template = template - np.mean(template)
    norm = np.linalg.norm(template)
    if norm > 1e-8:
        template = template / norm
    return template.astype(np.float32)


def max_template_correlation(template: np.ndarray, signal: np.ndarray) -> float:
    """Max correlation between template and signal (template assumed zero-mean & unit-norm)."""
    x = normalize_signal(signal)
    corr = np.correlate(x, template, mode='valid')
    return float(np.max(corr))


# ---------------------------------------------------------------
# Approach 2 features: hand-crafted
# ---------------------------------------------------------------

def extract_handcrafted_features(signal: np.ndarray,
                                 sr: float,
                                 n_bands: int = 32) -> Tuple[np.ndarray, List[str]]:
    x = normalize_signal(signal)
    feats: List[float] = []
    names: List[str] = []

    feats.append(float(np.mean(x)));      names.append('time_mean')
    feats.append(float(np.std(x)));       names.append('time_std')
    feats.append(float(np.max(x)));       names.append('time_max')
    feats.append(float(np.min(x)));       names.append('time_min')
    feats.append(float(np.ptp(x)));       names.append('time_ptp')
    feats.append(float(np.sqrt(np.mean(x ** 2)))); names.append('time_rms')

    abs_x = np.abs(x)
    feats.append(float(np.mean(abs_x)));  names.append('abs_mean')
    feats.append(float(np.std(abs_x)));   names.append('abs_std')
    feats.append(float(np.max(abs_x)));   names.append('abs_max')

    dx = np.diff(x)
    feats.append(float(np.mean(dx)));     names.append('diff_mean')
    feats.append(float(np.std(dx)));      names.append('diff_std')
    feats.append(float(np.max(dx)));      names.append('diff_max')
    feats.append(float(np.min(dx)));      names.append('diff_min')

    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mag = np.abs(np.fft.rfft(x))
    total = np.sum(mag) + 1e-12
    mag_norm = mag / total

    max_f = sr / 2.0
    band_edges = np.linspace(0.0, max_f, n_bands + 1)
    for i in range(n_bands):
        f0, f1 = band_edges[i], band_edges[i + 1]
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        e = float(np.sum(mag_norm[idx])) if len(idx) else 0.0
        feats.append(e)
        names.append(f'band_{i:02d}_{int(f0)}-{int(f1)}Hz')

    return np.array(feats, dtype=np.float32), names


def build_feature_matrix(signals: List[np.ndarray], sr: float, n_bands: int = 32) -> Tuple[np.ndarray, List[str]]:
    feat_list = []
    names_ref: List[str] = []
    for i, s in enumerate(signals):
        feats, names = extract_handcrafted_features(s, sr, n_bands=n_bands)
        feat_list.append(feats)
        if i == 0:
            names_ref = names
    return np.stack(feat_list, axis=0), names_ref


# ---------------------------------------------------------------
# Epoch-wise trainer for ML (SGD logistic regression)
# ---------------------------------------------------------------

def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train_sgd_logreg_curves(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            lr: float,
                            epochs: int,
                            batch_size: int,
                            seed: int = 42,
                            alpha: float = 1e-5) -> pd.DataFrame:
    """
    Train SGDClassifier(loss="log_loss") epoch-by-epoch and log:
      - train loss, val loss
      - train AUC,  val AUC
    """
    rng = np.random.default_rng(seed)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,            # regularization strength (higher -> smoother curves, less overfit)
        learning_rate="constant",
        eta0=lr,               # learning rate
        fit_intercept=True,
        random_state=seed,
        average=False
    )

    classes = np.array([0, 1], dtype=int)

    rows = []
    for ep in range(1, epochs + 1):
        # One epoch of mini-batch partial_fit
        for xb, yb in iter_minibatches(X_train, y_train, batch_size, rng):
            clf.partial_fit(xb, yb, classes=classes)

        # Probabilities for losses/AUC
        p_tr = clf.predict_proba(X_train)[:, 1]
        p_va = clf.predict_proba(X_val)[:, 1]

        tr_loss = log_loss(y_train, p_tr, labels=[0, 1])
        va_loss = log_loss(y_val, p_va, labels=[0, 1])

        # AUC can fail if a split is degenerate; handle safely
        try:
            tr_auc = roc_auc_score(y_train, p_tr)
        except Exception:
            tr_auc = np.nan
        try:
            va_auc = roc_auc_score(y_val, p_va)
        except Exception:
            va_auc = np.nan

        rows.append({
            "epoch": ep,
            "lr": lr,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_auc": tr_auc,
            "val_auc": va_auc
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Plotting (publication style, 400 dpi, big fonts)
# ---------------------------------------------------------------

def set_pub_style(font_size: int = 16):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "legend.fontsize": font_size - 2,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "axes.linewidth": 1.2,
        "figure.dpi": 120
    })


def save_curve_plot(df_all: pd.DataFrame,
                    y_col_train: str,
                    y_col_val: str,
                    title: str,
                    ylabel: str,
                    out_png: str,
                    out_pdf: str,
                    font_size: int = 16,
                    dpi: int = 400):
    set_pub_style(font_size=font_size)

    plt.figure(figsize=(7.5, 5.2))
    for lr in sorted(df_all["lr"].unique()):
        df = df_all[df_all["lr"] == lr].sort_values("epoch")
        plt.plot(df["epoch"], df[y_col_train], label=f"Train (lr={lr:g})")
        plt.plot(df["epoch"], df[y_col_val], linestyle="--", label=f"Val (lr={lr:g})")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, ncol=1)
    plt.tight_layout()

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------
# Feature-set builder
# ---------------------------------------------------------------

def build_features(feature_set: str,
                   signals: List[np.ndarray],
                   metas: List[Dict],
                   sr: float,
                   n_bands: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    feature_set:
      - "template"   -> 1D correlation feature
      - "hand"       -> handcrafted features
      - "combined"   -> concat(template, handcrafted)
    """
    feature_set = feature_set.lower().strip()

    if feature_set not in {"template", "hand", "combined"}:
        raise ValueError("feature_set must be one of: template, hand, combined")

    # handcrafted block
    X_hand, hand_names = build_feature_matrix(signals, sr, n_bands=n_bands)

    if feature_set == "hand":
        return X_hand, hand_names

    # template block
    template = build_red_template(signals, metas, sr)
    corr = np.array([max_template_correlation(template, s) for s in signals], dtype=np.float32).reshape(-1, 1)
    corr_names = ["template_maxcorr"]

    if feature_set == "template":
        return corr, corr_names

    # combined
    X = np.concatenate([corr, X_hand], axis=1)
    names = corr_names + hand_names
    return X, names


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def run(red_zip: str,
        not_red_zip: str,
        feature_set: str,
        epochs: int,
        lrs: List[float],
        batch_size: int,
        out_dir: str,
        seed: int,
        n_bands: int,
        alpha: float,
        font_size: int,
        dpi: int):

    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset...")
    signals, labels, metas, sr = load_dataset(red_zip, not_red_zip)
    print(f"Loaded {len(signals)} recordings (sr ~ {sr:.1f} Hz).")

    # Normalize + crop
    signals = [normalize_signal(s) for s in signals]
    signals, sig_len = crop_signals_to_min_length(signals)
    print(f"Cropped length: {sig_len} samples (~{sig_len/sr:.2f} s).")

    y = np.array(labels, dtype=int)

    # Build features for ALL samples
    print(f"Building features: {feature_set} ...")
    X_all, feat_names = build_features(feature_set, signals, metas, sr, n_bands=n_bands)
    print(f"Feature dim: {X_all.shape[1]}")

    # Train/test split (kept for fairness), then train/val split inside train
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=seed, stratify=y
    )

    X_train_full = X_all[idx_train]
    y_train_full = y[idx_train]
    X_test = X_all[idx_test]
    y_test = y[idx_test]

    idx_tr, idx_val = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full
    )

    X_train = X_train_full[idx_tr]
    y_train = y_train_full[idx_tr]
    X_val = X_train_full[idx_val]
    y_val = y_train_full[idx_val]

    # Standardize based on TRAIN only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Sweep LRs
    df_runs = []
    for lr in lrs:
        print(f"\nTraining SGD logistic regression: lr={lr:g}, epochs={epochs}, batch={batch_size}, alpha={alpha:g}")
        df = train_sgd_logreg_curves(
            X_train, y_train, X_val, y_val,
            lr=lr, epochs=epochs, batch_size=batch_size,
            seed=seed, alpha=alpha
        )
        df_runs.append(df)

        # Save per-run CSV
        csv_path = os.path.join(out_dir, f"curves_{feature_set}_lr{lr:g}.csv")
        df.to_csv(csv_path, index=False)
        print("Saved:", csv_path)

    df_all = pd.concat(df_runs, axis=0, ignore_index=True)

    # Save combined CSV
    all_csv = os.path.join(out_dir, f"curves_{feature_set}_ALL.csv")
    df_all.to_csv(all_csv, index=False)
    print("\nSaved:", all_csv)

    # Plot loss curves
    loss_png = os.path.join(out_dir, f"loss_{feature_set}.png")
    loss_pdf = os.path.join(out_dir, f"loss_{feature_set}.pdf")
    save_curve_plot(
        df_all,
        y_col_train="train_loss",
        y_col_val="val_loss",
        title=f"Training / Validation Loss ({feature_set} features)",
        ylabel="Log Loss",
        out_png=loss_png,
        out_pdf=loss_pdf,
        font_size=font_size,
        dpi=dpi
    )
    print("Saved:", loss_png, "and", loss_pdf)

    # Plot AUC curves
    auc_png = os.path.join(out_dir, f"auc_{feature_set}.png")
    auc_pdf = os.path.join(out_dir, f"auc_{feature_set}.pdf")
    save_curve_plot(
        df_all,
        y_col_train="train_auc",
        y_col_val="val_auc",
        title=f"Training / Validation AUC ({feature_set} features)",
        ylabel="ROC-AUC",
        out_png=auc_png,
        out_pdf=auc_pdf,
        font_size=font_size,
        dpi=dpi
    )
    print("Saved:", auc_png, "and", auc_pdf)

    # (Optional) quick final test evaluation for best val loss per lr
    # This is not for curves; just to sanity-check.
    print("\nNote: curves are based on train/val split only. Test evaluation can be added if needed.")


def parse_args():
    p = argparse.ArgumentParser(description="ML-only RED keyword detection: epoch curves + LR sweep (publication plots).")
    p.add_argument("--red_zip", type=str, default="data/Red_story_231125.zip")
    p.add_argument("--not_red_zip", type=str, default="data/not_RED_speaking.zip")

    p.add_argument("--feature_set", type=str, default="combined",
                   choices=["template", "hand", "combined"],
                   help="Which feature set to use for epoch curves.")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lrs", type=str, default="0.001,0.003,0.01,0.03",
                   help="Comma-separated learning rates (eta0) for SGDClassifier.")
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--alpha", type=float, default=1e-5,
                   help="L2 regularization strength for SGD (higher -> smoother, less overfit).")
    p.add_argument("--n_bands", type=int, default=32)

    p.add_argument("--out_dir", type=str, default="outputs/results_mlplots")
    p.add_argument("--seed", type=int, default=42)

    # publication plot controls
    p.add_argument("--font_size", type=int, default=16)
    p.add_argument("--dpi", type=int, default=400)

    return p.parse_args()


def main():
    args = parse_args()
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]

    run(
        red_zip=args.red_zip,
        not_red_zip=args.not_red_zip,
        feature_set=args.feature_set,
        epochs=args.epochs,
        lrs=lrs,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        seed=args.seed,
        n_bands=args.n_bands,
        alpha=args.alpha,
        font_size=args.font_size,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
