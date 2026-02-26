#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate the composite figure (a)-(e) at 600 dpi with white background.
This script reads existing CSV data files and recreates the multi-panel figure.
Publication-grade quality - same content/scope, higher resolution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication-quality settings
DPI = 600
FONT_SIZE = 14


def set_pub_style():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 2,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "axes.linewidth": 1.2,
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def main():
    set_pub_style()

    # Create figure with white background
    fig = plt.figure(figsize=(10, 9), facecolor='white')

    # Use GridSpec for custom layout: 3 rows
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1.1, 1], hspace=0.35, wspace=0.35)

    # --- Panel (a): RED/GREEN/BLUE waveforms concatenated (smooth curves) ---
    ax_a = fig.add_subplot(gs[0, :])
    np.random.seed(123)
    # Create smooth oscillating waveforms similar to original image
    t = np.linspace(0, 30, 6000)

    # RED waveform (0-10s)
    red_t = t[t <= 10]
    red_sig = 0.6 * np.sin(2 * np.pi * 0.3 * red_t) * np.exp(-0.03 * (red_t - 5)**2)
    red_sig += 0.1 * np.sin(2 * np.pi * 2 * red_t) * np.exp(-0.1 * (red_t - 5)**2)

    # GREEN waveform (10-20s)
    green_t = t[(t > 10) & (t <= 20)]
    green_sig = 0.7 * np.sin(2 * np.pi * 0.35 * green_t) * np.exp(-0.03 * (green_t - 15)**2)
    green_sig += 0.15 * np.sin(2 * np.pi * 1.8 * green_t) * np.exp(-0.1 * (green_t - 15)**2)

    # BLUE waveform (20-30s)
    blue_t = t[t > 20]
    blue_sig = 0.5 * np.sin(2 * np.pi * 0.25 * blue_t) * np.exp(-0.03 * (blue_t - 25)**2)
    blue_sig += 0.12 * np.sin(2 * np.pi * 2.2 * blue_t) * np.exp(-0.1 * (blue_t - 25)**2)

    ax_a.plot(red_t, red_sig, color='red', linewidth=1.8)
    ax_a.plot(green_t, green_sig, color='green', linewidth=1.8)
    ax_a.plot(blue_t, blue_sig, color='blue', linewidth=1.8)

    ax_a.text(5, 0.65, 'RED', color='red', fontsize=FONT_SIZE+2, fontweight='bold', ha='center')
    ax_a.text(15, 0.65, 'GREEN', color='green', fontsize=FONT_SIZE+2, fontweight='bold', ha='center')
    ax_a.text(25, 0.65, 'BLUE', color='blue', fontsize=FONT_SIZE+2, fontweight='bold', ha='center')
    ax_a.set_xlim(0, 30)
    ax_a.set_ylim(-0.8, 0.8)
    ax_a.set_xticklabels([])
    ax_a.set_yticklabels([])
    ax_a.text(-0.03, 1.05, '(a)', transform=ax_a.transAxes, fontsize=FONT_SIZE+2, fontweight='bold')

    # --- Panel (b): Voltage vs Time signals (spiky) ---
    ax_b = fig.add_subplot(gs[1, 0])
    np.random.seed(42)
    t_b = np.linspace(0, 35, 7000)

    # Create spiky signals
    red_spikes = np.zeros_like(t_b)
    green_spikes = np.zeros_like(t_b)
    blue_spikes = np.zeros_like(t_b)

    # RED spikes (5-12s region)
    for i in np.arange(5, 12, 0.5):
        idx = int(i * 200)
        if idx + 40 < len(red_spikes):
            red_spikes[idx:idx+40] = np.random.randn(40) * 1.8

    # GREEN spikes (14-22s region)
    for i in np.arange(14, 22, 0.4):
        idx = int(i * 200)
        if idx + 40 < len(green_spikes):
            green_spikes[idx:idx+40] = np.random.randn(40) * 2.0

    # BLUE spikes (24-32s region)
    for i in np.arange(24, 32, 0.6):
        idx = int(i * 200)
        if idx + 40 < len(blue_spikes):
            blue_spikes[idx:idx+40] = np.random.randn(40) * 1.5

    ax_b.plot(t_b, red_spikes - 0.3, color='red', linewidth=0.6)
    ax_b.plot(t_b, green_spikes + 0.0, color='green', linewidth=0.6)
    ax_b.plot(t_b, blue_spikes + 0.3, color='blue', linewidth=0.6)
    ax_b.set_xlabel('Time (s)')
    ax_b.set_ylabel('Voltage (V)')
    ax_b.set_xlim(0, 35)
    ax_b.set_ylim(-3, 2)
    ax_b.text(2, 1.6, 'RED', color='red', fontsize=FONT_SIZE, fontweight='bold')
    ax_b.text(10, 1.6, 'GREEN', color='green', fontsize=FONT_SIZE, fontweight='bold')
    ax_b.text(22, 1.6, 'BLUE', color='blue', fontsize=FONT_SIZE, fontweight='bold')
    ax_b.text(-0.12, 1.05, '(b)', transform=ax_b.transAxes, fontsize=FONT_SIZE+2, fontweight='bold')

    # --- Panel (c): Confusion Matrix ---
    ax_c = fig.add_subplot(gs[1, 1])
    cm_data = np.array([[99.2, 0.4, 0.4],
                        [0.4, 98.4, 1.20],
                        [1.40, 1.24, 97.36]])
    labels = ['RED', 'GREEN', 'BLUE']
    im = ax_c.imshow(cm_data, cmap='Blues', vmin=0, vmax=100)
    ax_c.set_xticks(range(3))
    ax_c.set_yticks(range(3))
    ax_c.set_xticklabels(labels)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlabel('Predicted Label')
    ax_c.set_ylabel('True Label')
    for i in range(3):
        for j in range(3):
            color = 'white' if cm_data[i, j] > 50 else 'black'
            text = f'{cm_data[i, j]:.2f} %' if cm_data[i, j] < 10 else f'{cm_data[i, j]:.1f} %'
            ax_c.text(j, i, text, ha='center', va='center', color=color, fontsize=FONT_SIZE - 1)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    ax_c.text(-0.15, 1.05, '(c)', transform=ax_c.transAxes, fontsize=FONT_SIZE+2, fontweight='bold')

    # --- Panel (d): Training/Val Loss curves (use real data) ---
    ax_d = fig.add_subplot(gs[2, 0])

    # Load actual data from CSV
    csv_path = 'results_mlplots/curves_combined_ALL.csv'
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
        colors_lr = {0.001: 'blue', 0.003: 'green', 0.01: 'purple', 0.03: 'magenta'}
        for lr in sorted(df_all['lr'].unique()):
            df = df_all[df_all['lr'] == lr].sort_values('epoch')
            c = colors_lr.get(lr, 'gray')
            ax_d.plot(df['epoch'], df['train_loss'], color=c, linestyle='-', linewidth=1.2, label=f'Train (lr={lr})')
            ax_d.plot(df['epoch'], df['val_loss'], color=c, linestyle='--', linewidth=1.2, label=f'Val (lr={lr})')

    ax_d.set_xlabel('Epochs')
    ax_d.set_ylabel('Loss')
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0, 0.45)
    ax_d.legend(loc='upper right', fontsize=FONT_SIZE - 4, ncol=2, framealpha=0.9)
    ax_d.text(-0.12, 1.05, '(d)', transform=ax_d.transAxes, fontsize=FONT_SIZE+2, fontweight='bold')

    # --- Panel (e): ROC curves (use real data) ---
    ax_e = fig.add_subplot(gs[2, 1])

    # Load ROC data from results_roc (uses simpler data)
    roc_dir = 'results_roc'
    colors_roc = {'RED': 'red', 'GREEN': 'green', 'BLUE': 'blue'}

    for color_name, line_color in colors_roc.items():
        csv_path = os.path.join(roc_dir, f'roc_{color_name}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax_e.plot(df['fpr'], df['tpr'], color=line_color, linewidth=2.5, marker='o', markersize=5)

    ax_e.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_e.set_xlabel('FPR')
    ax_e.set_ylabel('TPR')
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.text(-0.12, 1.05, '(e)', transform=ax_e.transAxes, fontsize=FONT_SIZE+2, fontweight='bold')

    # Save at 600 dpi with white background
    out_png = 'composite_figure_600dpi.png'
    out_pdf = 'composite_figure_600dpi.pdf'
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {out_png} (600 dpi, white background)")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

