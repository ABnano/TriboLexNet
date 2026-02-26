# TriboLexNet

![TriboLexNet acoustic banner](assets/tribolexnet_acoustic_banner.svg)

Minimal, publication-oriented repository for **RED keyword detection** from **triboelectric/piezoelectric acoustic recordings** using a **classical machine learning (Random Forest)** pipeline.

This repository is intentionally scoped to one reproducible workflow:

- one runnable entry script: `red_detect_rf_publi_rgb_pre.py`
- a small internal package for cleaner code organization
- only the **datasets required by this RED-only pipeline**
- no pre-generated plots, results, checkpoints, or experiment clutter

## Why only RED data is included

This repository currently targets a **RED keyword spotting** pipeline, not multi-class color classification.

The code in `red_detect_rf_publi_rgb_pre.py` (and the refactored package) only loads:

- `Red_story_231125.zip`
- `not_RED_speaking.zip`
- `5 times Red.zip`
- `Red 100x 10Hz.zip`
- `Red 100x 13Hz.zip`

The GREEN / BLUE / ORANGE / YELLOW / INDIGO / VIOLET ZIP archives are **not referenced by this pipeline** and were intentionally removed to keep the repository compact and professional.

## Repository Structure

```text
.
├── assets/
│   └── tribolexnet_acoustic_banner.svg
├── data/
│   ├── 5 times Red.zip
│   ├── Red 100x 10Hz.zip
│   ├── Red 100x 13Hz.zip
│   ├── Red_story_231125.zip
│   └── not_RED_speaking.zip
├── tribolexnet_red_rf/
│   ├── __init__.py
│   ├── cli.py
│   ├── core.py
│   └── plotting.py
├── red_detect_rf_publi_rgb_pre.py
├── .gitignore
└── README.md
```

## Refactor Summary (professionalized layout)

The original monolithic script was reorganized into a small package:

- `tribolexnet_red_rf/core.py`
  - ZIP loading
  - signal normalization / envelope / peak utilities
  - handcrafted feature extraction
  - Random Forest training and sliding-window inference
- `tribolexnet_red_rf/plotting.py`
  - uniform plotting style
  - confusion matrix / feature importance / histogram plots
  - example detection plots and combined 2x2 summary figure
- `tribolexnet_red_rf/cli.py`
  - command-line argument parsing
  - end-to-end pipeline orchestration
- `red_detect_rf_publi_rgb_pre.py`
  - thin backward-compatible entry point

This keeps the public command unchanged while making the codebase easier to maintain.

## Pipeline Overview

The RED RF pipeline performs:

1. Sentence-level dataset loading (`RED` vs `non-RED`) from ZIP archives
2. RED pulse duration estimation from single-word RED recordings
3. Segment extraction around RED pulse candidates
4. Handcrafted feature generation (time + spectral descriptors)
5. Segment-level Random Forest training
6. Sliding-window inference over sentence recordings
7. Clustering of high-probability windows into RED events
8. Plot export (PNG + PDF + SVG, titled and no-title variants)

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

Run the full pipeline with default paths (`data/` -> `outputs/`):

```bash
python red_detect_rf_publi_rgb_pre.py
```

Explicit paths:

```bash
python red_detect_rf_publi_rgb_pre.py --input_dir ./data --out_dir ./outputs
```

Save all waveform example detections:

```bash
python red_detect_rf_publi_rgb_pre.py --n_example_plots -1
```

Adjust publication plotting settings:

```bash
python red_detect_rf_publi_rgb_pre.py --dpi 600 --font_base 18 --topk 5
```

## Expected Inputs

The script expects the ZIP files to contain CSV recordings with:

- `;` as separator
- `,` as decimal separator
- 3 header lines skipped
- two columns interpreted as `Time_s` and `Current_nA`

## Outputs (generated locally, not tracked)

By default, outputs are written to:

- `outputs/plots/`
- `outputs/plots/examples/`
- `outputs/red_rf_summary_2x2.{png,pdf,svg}`
- `outputs/red_rf_summary_2x2_notitle.{png,pdf,svg}`

These are excluded via `.gitignore` to keep the repository clean.

## Reproducibility Notes

- Segment-level train/validation split uses fixed `random_state`
- Random Forest uses a fixed `random_state`
- The pipeline is deterministic except for external library version differences

## Scope and Next Steps

This repository is intentionally focused on a single RED keyword detection workflow.

If you later want a broader **multi-color acoustic command classification** repository (RED/GREEN/BLUE/etc.), create a separate pipeline and re-introduce those datasets with a distinct README and experiment structure.
