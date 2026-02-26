#!/usr/bin/env python3
"""CLI entry point for the TriboLexNet RED Random Forest pipeline."""

import argparse
import os
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from .core import (
    build_red_segments_from_single_files,
    build_training_sets,
    cluster_high_prob_windows,
    crop_signals_to_min_length,
    find_single_red_files,
    load_sentences,
    normalize_signal,
    sliding_window_probs_rf,
)
from .plotting import (
    combine_four_images_2x2_panel_dual,
    ensure_dir,
    plot_confusion_matrix_percent_dual,
    plot_detection_example_dual,
    plot_sentence_maxprob_hist_dual,
    set_plot_style,
)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random Forest keyword spotting pipeline for detecting the word 'RED' from piezoelectric acoustic recordings."
    )
    ap.add_argument("--input_dir", type=str, default="data",
                    help="Folder containing all required ZIP files.")
    ap.add_argument("--out_dir", type=str, default="outputs",
                    help="Where to write outputs.")
    ap.add_argument("--plots_subdir", type=str, default="plots",
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
