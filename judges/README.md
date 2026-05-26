# Qwen2.5-VL-72B-AWQ Judge

vLLM-based judge that scores generated meme transcreations on the same 5-metric rubric as the other VLM judges, using the official `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` checkpoint. Designed for single-GPU inference on a 48 GB-class card (e.g. L40S / A6000 Ada).

## Layout

```
judegs/
├── _common/
│   ├── vllm_judge.py        # VLLMJudge: loads the 72B-AWQ checkpoint via vLLM and scores one (orig, gen) pair
│   ├── pair_discovery.py    # discover_pairs() with per-generator rules (janus, gemma4, gemini, llava)
│   └── prompt.py            # PROMPT_TEMPLATE and format_prompt(direction)
├── run_generator.py         # CLI driver with streaming-write + resume; produces one CSV per (direction, variant)
└── slurm/                   # example SLURM scripts (adapt module / conda activation to your cluster)
```

## Requirements

- Python 3.11+
- vLLM 0.21.0+ (earlier versions handle Qwen2.5-VL multimodal differently)
- A CUDA-capable GPU with ~48 GB VRAM
- For first-time use, `nvcc` must be on PATH (the flashinfer JIT compiles a sampling kernel on first run)

The default `MODEL_PATH` is the Hugging Face model id `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`; vLLM will download (~40 GB) on first run. To use locally cached weights, set:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct-AWQ
```

## Image-pair layout

Pair discovery expects the following layout under `$MEMEXGEN_DATA_ROOT` (default `./data/for_gen`):

| Generator | Folder pattern | Original filename | Generated filename prefix |
|---|---|---|---|
| Janus     | `{ch2us,us2ch}_janus_image_pairs_{simple,structure}/NNN_<stem>/`     | starts with `meme_`           | `j_{c,u}_{si,st}_` |
| Gemma4    | `{ch2us,us2ch}_gemma4flux_image_pairs_{simple,structure}/NNN_<stem>/` | starts with `meme_`           | `g_{c,u}_{si,st}_` |
| Gemini    | `gemini/{US2CH,CN2US}/{1..100}/`                                       | anything other than generated | `Gemini_Generated_Image_*` |
| LLaVA     | (see `_discover_llava` in `pair_discovery.py`)                         | varies by direction           | varies by direction |

If your folder names include date suffixes (e.g. `..._20260413`), either rename or edit the `folder_name` entries in `run_generator.py::GENERATOR_CASES`.

## Running

```bash
cd judges/qwen2.5-vl-72b

# Optional: point at locally cached weights and your dataset root
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct-AWQ
export MEMEXGEN_DATA_ROOT=/path/to/for_gen

# One generator at a time (each loads the 72B once, ~3-5 min)
python run_generator.py --generator janus
python run_generator.py --generator gemma4
python run_generator.py --generator gemini
python run_generator.py --generator llava
```

On a multi-GPU cluster, submit the SLURM scripts independently so each lands on its own GPU:

```bash
sbatch slurm/janus.sbatch
sbatch slurm/gemma4.sbatch
sbatch slurm/gemini.sbatch
sbatch slurm/llava.sbatch
```

The driver streams writes (flush per row) and resumes from the last completed folder, so interrupted jobs can be re-submitted safely.

## Output

CSVs land under `results/` with one row per (direction, variant) pair:

```
index, folder, generated_image, original_image, evaluation_text, timestamp
```

`evaluation_text` contains the raw judge output. It includes a `## Meme Transcreation Evaluation` block with 5 numbered scores and an `**Overall Score**: X/5` line that downstream parsers can extract.

## Memory tuning notes

The settings in `vllm_judge.py` are calibrated for a single 48 GB GPU:

| Setting | Value | Why |
|---|---|---|
| `quantization` | `"awq"` | `awq_marlin` needs an extra ~1.8 GiB workspace; on a tight 48 GB card the marlin conversion OOMs |
| `gpu_memory_utilization` | `0.99` | Profiling peak ~46 GiB; lower this on a larger card |
| `max_model_len` | `4096` | Each prompt fits comfortably; raising this also raises KV-cache demand |
| `enforce_eager` | `True` | Skips CUDA-graph capture (~1 GiB saved) |
| `mm_processor_kwargs.max_pixels` | `1280 * 28 * 28` | Caps image patches; the default `12845056` triggers profiling OOM on 48 GB |

If you have more headroom, lowering `gpu_memory_utilization` and setting `enforce_eager=False` speeds inference up modestly.
