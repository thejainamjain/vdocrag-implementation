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

# Vision-token cap for the GENERATOR (Qwen2.5-VL). Its own default range is
# 4-16384 tokens PER IMAGE, uncapped -- with TOP_K=3 full-DPI page images
# that's easily tens of thousands of vision tokens in one prompt, which is
# what actually blew the VRAM budget during generate() (confirmed: the OOM
# traceback's failing allocation was inside attention over that sequence,
# not model loading). This range (256-1280 tokens/image) is transformers'
# own documented memory-saving recommendation -- still legible for reading
# text-heavy document scans, just no longer unbounded. The RETRIEVER
# (ColQwen2.5) doesn't need this: its checkpoint already ships a sane
# built-in cap (max_pixels tuned for ~768 patches, per the model card).
GENERATOR_MIN_PIXELS = 256 * 28 * 28
GENERATOR_MAX_PIXELS = 1280 * 28 * 28

# Memory-efficient attention kernel (SDPA -- fused, doesn't materialize the
# full O(n^2) attention score matrix the way "eager" does). Explicit rather
# than relying on the library's auto-selection default. flash_attention_2
# would be even more memory-efficient for this multi-image case per Qwen's
# own docs, but isn't set by default here since it requires a separate
# `pip install flash-attn` with a build step that isn't guaranteed to have a
# prebuilt wheel for Colab's exact CUDA/torch/Python combo -- worth trying
# by hand if you need more headroom, not defaulted to avoid breaking the
# install for people it doesn't have a wheel for.
ATTN_IMPLEMENTATION = "sdpa"


@dataclass
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    # float16, not bfloat16: T4 (Turing, compute capability 7.5) has no native
    # bf16 tensor cores -- those are Ampere+ (8.0+) only. bf16 still "works" on
    # a T4 but runs unaccelerated, which slows down both quantization at load
    # time and every forward pass after. float16 uses T4's actual fast path.
    # If you're running on an Ampere+ GPU (A100, etc.), bfloat16 is preferable
    # for its wider dynamic range -- change this back in that case.
    bnb_4bit_compute_dtype_name: str = "float16"
    bnb_4bit_use_double_quant: bool = True