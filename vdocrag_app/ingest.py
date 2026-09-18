"""PDF -> per-page PIL images. Requires poppler-utils installed on the system
(apt-get install poppler-utils in Colab -- see notebooks/run_in_colab.ipynb)."""

import logging
import os
import time
from typing import List

from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger("vdocrag")


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    start = time.time()
    images = convert_from_path(pdf_path, dpi=dpi)
    logger.info(f"Rasterized {os.path.basename(pdf_path)}: {len(images)} pages ({(time.time() - start) * 1000:.0f}ms)")
    return [img.convert("RGB") for img in images]


def save_page_images(images: List[Image.Image], out_dir: str, source_name: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(out_dir, f"{source_name}__page_{i}.png")
        img.save(path)
        paths.append(path)
    return paths
