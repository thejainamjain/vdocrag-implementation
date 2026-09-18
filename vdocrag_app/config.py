"""
Central config for Path B: ColQwen2.5 (retrieval) + Qwen2.5-VL-7B-Instruct (generation).

Both models are natively supported in `transformers` (no `trust_remote_code`),
which is precisely why Path B was chosen over Path A (Phi-3-vision) -- native
support means proper KV-cache generation (no input-mutation bug) and no custom
modeling-code interactions with bitsandbytes quantization to work around.
"""

from dataclasses import dataclass


# "Bigger" pair, tried first per the VRAM research:
#   Qwen2.5-VL-7B-Instruct 4-bit  ~4.5GB weights (verified: 8.3B params, native
#     transformers support -- much lighter 4-bit footprint than Phi-3-vision's
#     4.2B despite being a bigger model, because Phi-3-vision's custom
#     trust_remote_code path doesn't get the same quantization efficiency).
#   ColQwen2.5-3B (vidore/colqwen2.5-v0.2) 4-bit  ~2-3GB expected (3B backbone,
#     same native-support reasoning). Official vidore release, MIT/Apache-2.0.
# Combined weight footprint ~7-8GB, leaving a big margin on a 15GB T4 --
# NOT YET measured on real hardware from here, treat as "try this first."
RETRIEVER_MODEL_ID = "vidore/colqwen2.5-v0.2"
GENERATOR_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Smaller pair, kept "in queue" as a fallback if the bigger pair doesn't
# actually fit once measured on real hardware. Swap these in by changing the
# two IDs above -- nothing else in the codebase hardcodes model size.
FALLBACK_RETRIEVER_MODEL_ID = "vidore/colqwen2-v1.0"          # Qwen2-VL-2B backbone
FALLBACK_GENERATOR_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

TOP_K = 3  # paper's own reported sweet spot for k (unlike Path A, no known
# input-mutation bug forcing us down to k=1 as a diagnostic -- start at a
# reasonable value and adjust based on real results, not a workaround value.

MAX_NEW_TOKENS = 128


@dataclass
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype_name: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
