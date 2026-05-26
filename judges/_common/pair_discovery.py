"""Per-generator (original, generated) image-pair discovery.

Environment variables:
  MEMEXGEN_DATA_ROOT   Root of the generated-image dataset (default: ./data/for_gen).
                       Expected to contain one subfolder per (generator, direction, variant).
  MEMEXGEN_LLAVA_ROOT  Root of the LLaVA+FLUX pair output (default: ./data/llava13b/image).
"""

import os
from pathlib import Path
from typing import List, Tuple

FOR_GEN = Path(os.environ.get("MEMEXGEN_DATA_ROOT", "./data/for_gen"))
LLAVA_ROOT = Path(os.environ.get("MEMEXGEN_LLAVA_ROOT", "./data/llava13b/image"))


def discover_pairs(
    folder_name: str,
    generator: str,
    gen_prefix: str,
) -> List[Tuple[Path, Path, str]]:
    """Return sorted list of (original_path, generated_path, subfolder_name)."""
    if generator == "gemini":
        return _discover_gemini(folder_name)
    if generator == "llava":
        return _discover_llava(folder_name)
    return _discover_standard(folder_name, gen_prefix)


def _discover_standard(folder_name: str, gen_prefix: str) -> List[Tuple[Path, Path, str]]:
    root = FOR_GEN / folder_name
    pairs = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        orig = gen = None
        for f in sub.iterdir():
            name_lower = f.name.lower()
            if name_lower.startswith("meme_"):
                orig = f
            elif name_lower.startswith(gen_prefix.lower()):
                gen = f
        if orig and gen:
            pairs.append((orig, gen, sub.name))
        else:
            import sys
            print(f"WARNING: incomplete pair in {sub} (orig={orig}, gen={gen})", file=sys.stderr)
    return pairs


def _discover_llava(direction: str) -> List[Tuple[Path, Path, str]]:
    """First 100 pairs for the llava+flux pipeline."""
    import sys
    if direction == "ch2us":
        root = LLAVA_ROOT / "CH2US" / "image_pairs"
        pairs = []
        for sub in sorted(root.iterdir())[:100]:
            if not sub.is_dir():
                continue
            orig = gen = None
            for f in sub.iterdir():
                name = f.name.lower()
                if name.startswith("original_"):
                    orig = f
                elif "_flux." in name:
                    gen = f
            if orig and gen:
                pairs.append((orig, gen, sub.name))
            else:
                print(f"WARNING: incomplete pair in {sub}", file=sys.stderr)
        return pairs
    elif direction == "us2ch":
        root = LLAVA_ROOT / "US2CH"
        pairs = []
        for sub in sorted(root.iterdir())[:100]:
            if not sub.is_dir():
                continue
            orig = root / sub.name / "1_orig.png"
            gen  = root / sub.name / "2_flux.png"
            if orig.exists() and gen.exists():
                pairs.append((orig, gen, sub.name))
            else:
                print(f"WARNING: incomplete pair in {sub}", file=sys.stderr)
        return pairs
    else:
        raise ValueError(f"Unknown llava direction: {direction}")


def _discover_gemini(folder_name: str) -> List[Tuple[Path, Path, str]]:
    root = FOR_GEN / folder_name

    def sort_key(d: Path):
        try:
            return (0, int(d.name))
        except ValueError:
            return (1, d.name)

    pairs = []
    for sub in sorted(root.iterdir(), key=sort_key):
        if not sub.is_dir():
            continue
        orig = gen = None
        for f in sub.iterdir():
            if "gemini_generated_image" in f.name.lower():
                gen = f
            else:
                orig = f
        if orig and gen:
            pairs.append((orig, gen, sub.name))
        else:
            import sys
            print(f"WARNING: incomplete pair in {sub} (orig={orig}, gen={gen})", file=sys.stderr)
    return pairs
