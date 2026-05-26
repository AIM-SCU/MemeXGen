# Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models

This repository contains the dataset and code accompanying the paper.

---

## Overview

Memes don’t just translate — they must be *transcreated*.

We study **cross-cultural meme transcreation** between Chinese and US cultures, a multimodal generation task that preserves communicative intent and humor while adapting culture-specific references, imagery, and stylistic conventions.

Our work introduces:
- A large-scale bidirectional Chinese ↔ US meme transcreation dataset
- A hybrid vision–language framework for multimodal adaptation
- A systematic analysis of how humor transfers (or fails to transfer) across cultures

<p align="center">
  <img src="overview2.png" alt="MemeXGen Overview" width="55%">
</p>

---

## Dataset

The image pairs and corresponding annotations can be accessed here:


🔗 https://drive.google.com/drive/folders/1cOXV2KaRvwJOIf3z28q6Z_6v6MGRJ9yq?usp=drive_link

---

## Repository structure

```
MemeXGen-main/
├── README.md
├── LICENSE
├── overview2.png
├── meme_eva/
│   └── run.py    LLaVA-v1.6-13B transcreation pipeline (generation)
├── judges/
│   └── qwen2.5-vl-72b/                  Qwen2.5-VL-72B-AWQ judge (vLLM); see folder README
│       ├── README.md
│       ├── _common/                     vllm_judge.py, pair_discovery.py, prompt.py
│       ├── run_generator.py             CLI driver; one CSV per direction × variant
│       └── slurm/                       Example SLURM scripts (adapt to your cluster)
└── data/
    ├── CH2US/, US2CH/                   Sample image pairs (CH→US, US→CH)
    ├── human_evaluations/               3 native-speaker evaluators (xlsx, one per judge)
    ├── llm_evaluations/                 6 VLM judges × LLaVA+FLUX outputs (xlsx)
    ├── baseline_evaluations/            7 VLM judges × {Gemini, Gemma4+FLUX, Janus-Pro-7B} (xlsx)
    │   ├── Gemma4/
    │   ├── Genimi/                      Gemini
    │   └── Janus-Pro-7B/
    └── evaluation_output_test.xlsx
```

The full meme dataset lives on Google Drive (link above); the `data/` folder here ships a small sample for inspection.

---

## Installation

Python 3.11+ is recommended. The repository has two independent runnable components with disjoint requirements; install only what you need.

**Common:**
```bash
pip install pillow pandas openpyxl
```

**For the transcreation pipeline (`meme_eva/`):**
```bash
pip install "torch>=2.1" "transformers>=4.39"
```
GPU recommended (LLaVA-v1.6-13B fits comfortably on a 24 GB-class card).

**For the Qwen2.5-VL-72B judge (`judges/qwen2.5-vl-72b/`):**
```bash
pip install "vllm>=0.21"
```
Single-GPU inference; requires a ~48 GB-class GPU (e.g. L40S / A6000 Ada). See [`judges/qwen2.5-vl-72b/README.md`](judges/qwen2.5-vl-72b/README.md) for memory tuning notes.

---

## How to run

### 1. Generate transcreations (LLaVA + FLUX pipeline)

`meme_eva/full_test_check_batch_roll.py` runs Stage 1 (cultural analysis + caption + image-generation instructions) over a dataset CSV. Configure the dataset / images directory via environment variables or CLI flags:

```bash
export MEMEXGEN_DATASET=/path/to/labeled_data.csv
export MEMEXGEN_IMAGES_DIR=/path/to/original_memes/

python meme_eva/full_test_check_batch_roll.py \
    --samples 300 \
    --seed 42 \
    --output test_results
```

The script supports checkpointing and resumes from the latest saved state if interrupted. The FLUX image-generation stage is run separately (see paper §4).

### 2. Score outputs with the Qwen2.5-VL-72B judge

The 7th judge can be run over any system's outputs (LLaVA+FLUX, Gemini, Gemma4+FLUX, Janus-Pro-7B). Layout the generated images per generator (see [`judges/qwen2.5-vl-72b/README.md`](judges/qwen2.5-vl-72b/README.md) for the expected folder pattern), then:

```bash
cd judges/qwen2.5-vl-72b

# Optional: point at locally cached weights and dataset root
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct-AWQ
export MEMEXGEN_DATA_ROOT=/path/to/for_gen

python run_generator.py --generator janus     # also: gemma4, gemini, llava
```

On a SLURM cluster, submit `slurm/{janus,gemma4,gemini,llava}.sbatch` (adapt `module load` / `conda activate` to your environment). The driver streams writes and resumes from the last completed folder.

### 3. Inspect evaluation outputs

The XLSX files under `data/llm_evaluations/`, `data/baseline_evaluations/`, and `data/human_evaluations/` are pre-computed and self-contained — open directly in Excel / Google Sheets, or load with `pandas.read_excel(...)`.

---

## Output structure

All evaluation files (both pre-computed XLSX in `data/` and CSVs written by `judges/qwen2.5-vl-72b/run_generator.py`) share the same schema, with one sheet named **`Meme Evaluations`** and 14 columns:

| # | Column | Description |
|---|---|---|
| 1 | `index` | 0-based row index |
| 2 | `folder` | Subfolder name of the meme pair |
| 3 | `generated_image` | Basename of the generated meme image |
| 4 | `original_image` | Basename of the source meme image |
| 5 | `evaluation_text` | Raw judge output (multi-paragraph rubric response) |
| 6 | `timestamp` | ISO-8601 timestamp of when the judgement was produced |
| 7 | `caption_quality` | 1–5, parsed from `evaluation_text` |
| 8 | `image_quality` | 1–5 |
| 9 | `synergy` | 1–5 (image-caption synergy) |
| 10 | `cultural_fit` | 1–5 |
| 11 | `intent_preservation` | 1–5 |
| 12 | `overall_score` | Floating-point average reported by the judge |
| 13 | `offensiveness` | `Not Offensive` / `Potentially Offensive` / `Clearly Offensive` |
| 14 | `usability` | `High` / `Medium` / `Low` |

Some rows may have empty parsed-score cells when the model's free-text response did not match the rubric format; the raw `evaluation_text` is retained in those cases.

The driver script (`run_generator.py`) writes CSVs to `judges/qwen2.5-vl-72b/results/` with the first 6 columns; the remaining 8 columns are derived from `evaluation_text` by downstream parsing.

