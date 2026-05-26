"""Qwen2.5-VL-72B-AWQ judge wrapper using vLLM (single-GPU inference).

Environment variables:
  QWEN_MODEL_PATH  Path to local AWQ weights, or HF model id (defaults to
                   "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", which auto-downloads).

Memory configuration is tuned for a single 48GB GPU (e.g. L40S / Ada-class).
If you OOM on a different card, reduce gpu_memory_utilization or max_model_len.
"""

import os
import base64
from io import BytesIO

# Must be set before torch is initialised; allows the allocator to grow existing
# segments rather than failing when a contiguous block can't be found. Helps
# when AWQ activation memory peaks during profiling.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
)


def _pil_to_b64url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLLMJudge:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_PATH,
            quantization="awq",              # plain awq fits on a 48GB card; awq_marlin needs ~1.8 GiB extra workspace
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.99,     # tight on a single 48GB GPU; lower if you have headroom
            tensor_parallel_size=1,          # single GPU (no NVLink assumed)
            limit_mm_per_prompt={"image": 2},
            trust_remote_code=True,
            enforce_eager=True,              # disables CUDA-graph capture to save ~1 GiB
            mm_processor_kwargs={
                # Qwen2.5-VL default max_pixels = 12845056 (16384 patches) — overkill for memes
                # and triggers profiling OOM on a 48GB card. 1280 patches (~1000x1000 px) is
                # more than enough for meme images.
                "max_pixels": 1280 * 28 * 28,
            },
        )
        self.sp = SamplingParams(temperature=0.0, max_tokens=1024)

    def judge(self, orig_path: str, gen_path: str, prompt: str) -> str:
        orig = Image.open(orig_path).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _pil_to_b64url(orig)}},
                {"type": "image_url", "image_url": {"url": _pil_to_b64url(gen)}},
                {"type": "text", "text": prompt},
            ],
        }]
        outputs = self.llm.chat(messages, sampling_params=self.sp)
        return outputs[0].outputs[0].text
