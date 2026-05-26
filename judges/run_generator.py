"""Run the Qwen2.5-VL-72B-AWQ judge over a generator's image pairs.

Usage:
  python run_generator.py --generator janus
  python run_generator.py --generator gemma4
  python run_generator.py --generator gemini
  python run_generator.py --generator llava

Each generator produces 2-4 output CSVs (one per direction x variant), with
columns: index, folder, generated_image, original_image, evaluation_text, timestamp.
CSV writes are flushed every row; reruns resume from the last completed folder.
"""

import argparse
import csv
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common.pair_discovery import discover_pairs
from _common.prompt import format_prompt
from _common.vllm_judge import VLLMJudge

RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = Path(__file__).parent / "logs"

# (direction, variant, folder_name_under_for_gen, gen_prefix)
# Folder names assume the layout shipped with the released image dataset.
GENERATOR_CASES = {
    "janus": [
        ("ch2us", "simple",    "ch2us_janus_image_pairs_simple",    "j_c_si_"),
        ("ch2us", "structure", "ch2us_janus_image_pairs_structure", "j_c_st_"),
        ("us2ch", "simple",    "us2ch_janus_image_pairs_simple",    "j_u_si_"),
        ("us2ch", "structure", "us2ch_janus_image_pairs_structure", "j_u_st_"),
    ],
    "gemma4": [
        ("ch2us", "simple",    "ch2us_gemma4flux_image_pairs_simple",    "g_c_si_"),
        ("ch2us", "structure", "ch2us_gemma4flux_image_pairs_structure", "g_c_st_"),
        ("us2ch", "simple",    "us2ch_gemma4flux_image_pairs_simple",    "g_u_si_"),
        ("us2ch", "structure", "us2ch_gemma4flux_image_pairs_structure", "g_u_st_"),
    ],
    "gemini": [
        ("us2ch", None, "gemini/US2CH", None),
        ("cn2us", None, "gemini/CN2US", None),
    ],
    "llava": [
        ("ch2us", "structure", "ch2us", None),
        ("us2ch", "structure", "us2ch", None),
    ],
}


def csv_path_for(generator: str, direction: str, variant) -> Path:
    if generator == "gemini":
        return RESULTS_DIR / f"qwen72b_meme_results_{direction}_gemini.csv"
    if generator == "llava":
        return RESULTS_DIR / f"qwen72b_meme_results_{direction}_{variant}_llava.csv"
    return RESULTS_DIR / f"qwen72b_meme_results_{direction}_{variant}_{generator}.csv"


def get_resume_state(path: Path):
    """Return (processed_folder_set, next_index)."""
    if not path.exists():
        return set(), 0
    processed = set()
    max_idx = -1
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row.get("index", -1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
            eval_text = row.get("evaluation_text", "").strip()
            if eval_text and not eval_text.startswith("ERROR:"):
                processed.add(row["folder"])
    return processed, max_idx + 1


def run_case(judge: VLLMJudge, generator: str, direction: str, variant, folder_name: str, gen_prefix, log):
    out_path = csv_path_for(generator, direction, variant)
    processed, next_idx = get_resume_state(out_path)

    pairs = discover_pairs(folder_name, generator, gen_prefix)
    pending = [(o, g, name) for o, g, name in pairs if name not in processed]

    log.info(f"[{generator}/{direction}/{variant}] {len(pairs)} pairs total, "
             f"{len(processed)} already done, {len(pending)} to process -> {out_path.name}")

    if not pending:
        log.info("Nothing to do - skipping.")
        return

    prompt = format_prompt(direction)
    file_exists = out_path.exists()

    with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["index", "folder", "generated_image", "original_image",
                             "evaluation_text", "timestamp"])

        for i, (orig_path, gen_path, folder) in enumerate(pending):
            try:
                eval_text = judge.judge(str(orig_path), str(gen_path), prompt)
                log.info(f"  [{next_idx}] {folder} - OK ({len(eval_text)} chars)")
            except Exception:
                eval_text = f"ERROR: {traceback.format_exc()}"
                log.error(f"  [{next_idx}] {folder} - FAILED\n{eval_text}")

            writer.writerow([
                next_idx,
                folder,
                gen_path.name,
                orig_path.name,
                eval_text,
                datetime.now().isoformat(),
            ])
            f.flush()
            next_idx += 1

    log.info(f"[{generator}/{direction}/{variant}] Done. CSV: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, choices=list(GENERATOR_CASES))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    log_file = LOGS_DIR / f"{args.generator}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"=== run_generator START  generator={args.generator} ===")

    judge = VLLMJudge()
    log.info("Model loaded.")

    for direction, variant, folder_name, gen_prefix in GENERATOR_CASES[args.generator]:
        run_case(judge, args.generator, direction, variant, folder_name, gen_prefix, log)

    log.info(f"=== run_generator DONE  generator={args.generator} ===")


if __name__ == "__main__":
    main()
