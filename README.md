# TriboLexNet

![TriboLexNet acoustic banner](assets/tribolexnet_acoustic_banner.svg)

Publication-oriented repository for acoustic keyword and color-signal experiments on triboelectric/piezoelectric recordings. It includes a cleaned RED keyword Random Forest pipeline, an RGB ROC analysis script, and an ML learning-curve script.

This repository is intentionally kept compact and reproducible:

- a primary RED keyword spotting workflow (`red_detect_rf_publi_rgb_pre.py`)
- a small internal package for cleaner code organization (`tribolexnet_red_rf/`)
- two additional analysis scripts restored from your earlier working directory:
  - `red_keyword_detection_mlplots.py` (learning curves / AUC curves)
  - `roc_colors_ml.py` (BLUE/RED/GREEN ROC experiments)
- only the datasets required by these scripts (no unrelated color ZIPs)
- no pre-generated plots, results, checkpoints, or experiment clutter

## Included Data Scope

The main pipeline is RED-only, but the additional ROC script (`roc_colors_ml.py`) needs BLUE, RED, and GREEN single-word recordings.

Included ZIPs in `data/` cover all currently tracked scripts:

- `Red_story_231125.zip`
- `not_RED_speaking.zip`
- `5 times Red.zip`
- `Red 100x 10Hz.zip`
- `Red 100x 13Hz.zip`
- `blue 100x 10Hz.zip`
- `Green 100x 13Hz.zip`

Still intentionally excluded (not required by the current scripts):

- ORANGE / YELLOW / INDIGO / VIOLET ZIP datasets

## Repository Structure

```text
.
|- assets/
|  `- tribolexnet_acoustic_banner.svg
|- data/
|  |- 5 times Red.zip
|  |- Red 100x 10Hz.zip
|  |- Red 100x 13Hz.zip
|  |- Red_story_231125.zip
|  |- not_RED_speaking.zip
|  |- blue 100x 10Hz.zip
|  `- Green 100x 13Hz.zip
|- tribolexnet_red_rf/
|  |- __init__.py
|  |- cli.py
|  |- core.py
|  `- plotting.py
|- red_detect_rf_publi_rgb_pre.py
|- red_keyword_detection_mlplots.py
|- roc_colors_ml.py
|- .gitignore
`- README.md
```

## Refactor Summary (professionalized layout)

The original monolithic RED RF script was reorganized into a small package:

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

## Included Scripts

### 1) RED keyword Random Forest pipeline (main)

- Entry point: `red_detect_rf_publi_rgb_pre.py`
- Refactored implementation: `tribolexnet_red_rf/`
- Purpose: RED keyword spotting with handcrafted features + Random Forest + publication-ready plots

### 2) ML learning curves / AUC curves (RED vs not-RED)

- Script: `red_keyword_detection_mlplots.py`
- Purpose: train/validation loss and AUC curves across learning-rate sweeps (SGDClassifier, no neural nets)
- Default inputs:
  - `data/Red_story_231125.zip`
  - `data/not_RED_speaking.zip`

### 3) RGB ROC experiments (BLUE / RED / GREEN vs negative)

- Script: `roc_colors_ml.py`
- Purpose: generate ROC curves and AUC tables for BLUE/RED/GREEN one-vs-negative experiments
- Default inputs:
  - `data/blue 100x 10Hz.zip`
  - `data/Red 100x 10Hz.zip`
  - `data/Green 100x 13Hz.zip`
  - `data/not_RED_speaking.zip`
- Note: this script uses PyTorch

## Pipeline Overview (main RED RF workflow)

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

Install core dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Optional dependency for `roc_colors_ml.py`:

```bash
pip install torch
```

## Usage

Run the main RED RF pipeline with default paths (`data/` -> `outputs/`):

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

### Additional Script Examples

Learning curves / LR sweep:

```bash
python red_keyword_detection_mlplots.py
```

RGB ROC experiments:

```bash
python roc_colors_ml.py
```

## Expected Inputs

All tracked scripts expect ZIP-contained CSV recordings with:

- `;` as separator
- `,` as decimal separator
- 3 header lines skipped
- two columns interpreted as `Time_s` and `Current_nA`

## Outputs (generated locally, not tracked)

By default, outputs are written under `outputs/` (for the scripts patched in this repo), such as:

- `outputs/plots/`
- `outputs/plots/examples/`
- `outputs/results_mlplots/`
- `outputs/results_roc_nn/`
- `outputs/red_rf_summary_2x2.{png,pdf,svg}`

These are excluded via `.gitignore` to keep the repository clean.

## Reproducibility Notes

- Segment-level train/validation split uses fixed `random_state`
- Random Forest uses a fixed `random_state`
- Other scripts also set explicit seeds where applicable
- Exact results may vary slightly across library versions / hardware

## Scope

The repository now contains a curated subset of your original work:

- RED RF detection pipeline (cleaned and modularized)
- RED learning-curve analysis script
- RGB ROC analysis script

It intentionally excludes unrelated scripts, checkpoints, and generated results to stay maintainable and presentation-ready.
