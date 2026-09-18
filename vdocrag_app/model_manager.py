"""
Loads both models once (native `transformers`/`colpali_engine` classes, no
`trust_remote_code`), each 4-bit quantized via bitsandbytes.

Unlike Path A (Phi-3-vision), these are two genuinely different model
architectures (ColQwen2.5 is a Qwen2.5-VL-3B backbone + a ColBERT-style
projection head; the generator is a separate Qwen2.5-VL-7B) -- there is no
single shared base model to hot-swap LoRA adapters on top of, so both are
just loaded independently. This is expected to still comfortably fit a T4
given the ~7-8GB combined 4-bit weight estimate (see config.py) -- NOT yet
confirmed on real hardware.
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

        retriever_id = config.FALLBACK_RETRIEVER_MODEL_ID if self.use_fallback else config.RETRIEVER_MODEL_ID
        generator_id = config.FALLBACK_GENERATOR_MODEL_ID if self.use_fallback else config.GENERATOR_MODEL_ID

        logger.info(f"Loading retriever: {retriever_id}")
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.retriever_model = ColQwen2_5.from_pretrained(
            retriever_id,
            quantization_config=_bnb_config(),
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.retriever_processor = ColQwen2_5_Processor.from_pretrained(retriever_id)
        logger.info("Retriever loaded.")

        logger.info(f"Loading generator: {generator_id}")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.generator_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            generator_id,
            quantization_config=_bnb_config(),
            torch_dtype=torch.bfloat16,
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
