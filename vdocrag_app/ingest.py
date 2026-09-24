"""PDF/image -> per-page PIL images. PDF rasterization requires poppler-utils
installed on the system (apt-get install poppler-utils in Colab -- see
notebooks/run_in_colab.ipynb). Plain image files (jpg/png/...) don't need
poppler at all -- they're just opened directly as a single "page"."""

import logging
import os
import time
from typing import List

from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger("vdocrag")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def file_to_images(path: str) -> List[Image.Image]:
    """Dispatches on extension: PDFs get rasterized page-by-page; a plain
    image file is treated as a single-page document. Same return shape
    (List[Image.Image]) either way, so callers don't need to care which
    kind of file they got."""
    if is_image_file(path):
        return image_to_pages(path)
    return pdf_to_images(path)


def image_to_pages(image_path: str) -> List[Image.Image]:
    start = time.time()
    img = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded {os.path.basename(image_path)} as a single page ({(time.time() - start) * 1000:.0f}ms)")
    return [img]


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