"""
Generation via Qwen2.5-VL-7B-Instruct's standard, documented usage pattern:
chat-template messages with image content blocks + qwen_vl_utils.process_vision_info.

Deliberately using the model's own default generation settings (use_cache
left at its default of True) -- unlike Path A's Phi-3-vision, this is a
natively-supported transformers model, so there's no known input_ids-mutation
issue forcing use_cache=False, and no reason to disable the normal, faster,
correct KV-cache codepath.
"""

import logging
from typing import List

import torch
from PIL import Image

logger = logging.getLogger("vdocrag")


class Generator:
    def __init__(self, model_manager, max_new_tokens: int = 128):
        self._mm = model_manager
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def answer(self, question: str, images: List[Image.Image]) -> str:
        from qwen_vl_utils import process_vision_info

        model, processor = self._mm.use_generator()

        if not images:
            return "(no pages retrieved -- index a document first)"

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": f"{question}\nAnswer concisely, based only on the image(s) shown."})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        # Strip the prompt tokens back off, keeping only the newly generated answer.
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0].strip()
        logger.info(f"Generated answer ({len(images)} images, {len(answer)} chars)")
        return answer
