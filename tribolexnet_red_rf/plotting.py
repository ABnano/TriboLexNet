#!/usr/bin/env python3
"""Plotting utilities for the TriboLexNet RED Random Forest pipeline."""

import os
from typing import Dict, Any, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.ensemble import RandomForestClassifier

# Color policy: ONLY RED / GREEN / BLUE for plotted elements
C_RED = "red"
C_GREEN = "green"
C_BLUE = "blue"

# Discrete 3-color map for confusion matrices (0-100%)
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
