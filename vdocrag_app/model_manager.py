"""
Loads both models once (native `transformers`/`colpali_engine` classes, no
`trust_remote_code`), each 4-bit quantized via bitsandbytes.

Unlike Path A (Phi-3-vision), these are two genuinely different model
architectures (ColQwen2.5 is a Qwen2.5-VL-3B backbone + a ColBERT-style
projection head; the generator is a separate Qwen2.5-VL-7B) -- there is no
single shared base model to hot-swap LoRA adapters on top of, so both are
just loaded independently.

The primary and fallback pairs are architecturally distinct, not just
differently-sized: ColQwen2.5/Qwen2.5-VL vs. ColQwen2/Qwen2-VL are separate
model classes in colpali_engine/transformers with structurally different
vision towers, so `load()` picks the matching class pair per `use_fallback`
rather than hardcoding one pair's classes for both. The ~7-8GB combined
4-bit weight estimate for the primary pair (see config.py) was optimistic --
confirmed on real free-tier-Colab-T4 hardware to be slow to download/quantize
(~20GB of full-precision weights fetched before quantization even starts).
If you're VRAM- or time-constrained, prefer the fallback pair outright rather
than treating it as a rare escape hatch.
"""

import logging

from . import config

logger = logging.getLogger("vdocrag")


def _bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    qc = config.QuantConfig()
    compute_dtype = getattr(torch, qc.bnb_4bit_compute_dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=qc.load_in_4bit,
        bnb_4bit_quant_type=qc.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qc.bnb_4bit_use_double_quant,
    )


class ModelManager:
    """
    Loaded once (e.g. via st.cache_resource in app.py) and shared across the
    Streamlit session. Exposes the two model+processor pairs; retriever.py and
    generator.py each pull from here rather than loading anything themselves.
    """

    def __init__(self, use_fallback: bool = False):
        self.use_fallback = use_fallback
        self.retriever_model = None
        self.retriever_processor = None
        self.generator_model = None
        self.generator_processor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self

        import torch

        if not torch.cuda.is_available():
            reason = "torch was not built with CUDA support" if not torch.backends.cuda.is_built() \
                else "no CUDA GPU is visible to this process"
            raise RuntimeError(
                f"No usable GPU found ({reason}). In Colab: Runtime > Change runtime "
                "type > Hardware accelerator > select a GPU (e.g. T4) > Save, then "
                "Runtime > Restart session and re-run all cells from the top. This "
                "app loads two multi-billion-parameter models 4-bit-quantized via "
                "bitsandbytes, which requires CUDA -- it will not run on CPU."
            )

        retriever_id = config.FALLBACK_RETRIEVER_MODEL_ID if self.use_fallback else config.RETRIEVER_MODEL_ID
        generator_id = config.FALLBACK_GENERATOR_MODEL_ID if self.use_fallback else config.GENERATOR_MODEL_ID
        compute_dtype = getattr(torch, config.QuantConfig().bnb_4bit_compute_dtype_name)

        logger.info(f"Loading retriever: {retriever_id}")
        if self.use_fallback:
            # vidore/colqwen2-v1.0 is a ColQwen2 (Qwen2-VL backbone) checkpoint --
            # NOT ColQwen2.5. Loading it with the ColQwen2_5 class produces
            # missing/unexpected-key or shape-mismatch errors: the two
            # architectures' vision towers differ structurally.
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            retriever_cls, retriever_processor_cls = ColQwen2, ColQwen2Processor
        else:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            retriever_cls, retriever_processor_cls = ColQwen2_5, ColQwen2_5_Processor

        self.retriever_model = retriever_cls.from_pretrained(
            retriever_id,
            quantization_config=_bnb_config(),
            torch_dtype=compute_dtype,
            device_map="cuda:0",
        ).eval()
        self.retriever_processor = retriever_processor_cls.from_pretrained(retriever_id)
        logger.info("Retriever loaded.")

        logger.info(f"Loading generator: {generator_id}")
        from transformers import AutoProcessor
        if self.use_fallback:
            # Qwen/Qwen2-VL-2B-Instruct needs Qwen2VLForConditionalGeneration --
            # a genuinely different transformers class/config from Qwen2.5-VL's.
            from transformers import Qwen2VLForConditionalGeneration
            generator_cls = Qwen2VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            generator_cls = Qwen2_5_VLForConditionalGeneration

        self.generator_model = generator_cls.from_pretrained(
            generator_id,
            quantization_config=_bnb_config(),
            torch_dtype=compute_dtype,
            device_map="cuda:0",
        ).eval()
        self.generator_processor = AutoProcessor.from_pretrained(generator_id)
        logger.info("Generator loaded.")

        self._loaded = True
        return self

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("ModelManager.load() must be called before use.")

    def use_retriever(self):
        self._require_loaded()
        return self.retriever_model, self.retriever_processor

    def use_generator(self):
        self._require_loaded()
        return self.generator_model, self.generator_processor