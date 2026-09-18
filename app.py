"""
Streamlit app -- Path B replacement for the earlier Gradio UI.

Run locally: `streamlit run app.py`
Run in Colab: see notebooks/run_in_colab.ipynb (runs this as a background
process, tunneled out via localtunnel).
"""

import logging
import os

import streamlit as st
from PIL import Image

from vdocrag_app import ingest
from vdocrag_app.config import TOP_K
from vdocrag_app.generator import Generator
from vdocrag_app.index import MultiVectorIndex, PageRecord
from vdocrag_app.model_manager import ModelManager
from vdocrag_app.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vdocrag")

# Persisted across Colab sessions via Google Drive -- see notebook for mount step.
DATA_DIR = os.environ.get("VDOCRAG_DATA_DIR", "./vdocrag_data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

st.set_page_config(page_title="VDocRAG demo (Path B)", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run only, ~1-2 min)...")
def get_model_manager(use_fallback: bool) -> ModelManager:
    mm = ModelManager(use_fallback=use_fallback)
    mm.load()
    return mm


def get_index() -> MultiVectorIndex:
    if "index" not in st.session_state:
        st.session_state.index = MultiVectorIndex.load(INDEX_DIR)
    return st.session_state.index


st.title("VDocRAG demo -- Path B")
st.caption(
    "Retrieval: ColQwen2.5-3B (multi-vector / ColBERT-style). "
    "Generation: Qwen2.5-VL-7B-Instruct. Both natively supported in `transformers`."
)

use_fallback = st.sidebar.checkbox(
    "Use smaller fallback models (Qwen2-VL-2B + ColQwen2-2B)",
    value=False,
    help="Switch to this if the bigger pair doesn't fit your GPU's VRAM.",
)

mm = get_model_manager(use_fallback)
retriever = Retriever(mm)
generator = Generator(mm)
index = get_index()

tab_upload, tab_ask = st.tabs(["Upload & Index", "Ask"])

with tab_upload:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"])

    if uploaded is not None and st.button("Index this PDF"):
        source_name = uploaded.name
        tmp_path = os.path.join(DATA_DIR, source_name)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Rasterizing PDF..."):
            pages = ingest.pdf_to_images(tmp_path)
            image_paths = ingest.save_page_images(pages, IMAGES_DIR, source_name)

        index.remove_source(source_name)  # re-indexing replaces previous pages for this file

        progress = st.progress(0.0, text="Encoding pages...")
        records = []
        for i, (img, path) in enumerate(zip(pages, image_paths)):
            emb = retriever.encode_document(img)
            records.append(
                PageRecord(doc_id=f"{source_name}::page_{i}", source=source_name, page_number=i, image_path=path, embedding=emb)
            )
            progress.progress((i + 1) / len(pages), text=f"Encoding pages... ({i + 1}/{len(pages)})")

        index.add_batch(records)
        index.save(INDEX_DIR)
        st.success(f"Indexed {len(records)} pages from '{source_name}'.")

    st.divider()
    st.write(f"**Currently indexed pages:** {len(index.records)}")
    if index.records:
        sources = sorted(set(r.source for r in index.records))
        st.write(", ".join(sources))

with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input("Question")
    top_k = st.slider("Number of pages to retrieve (k)", 1, 5, TOP_K)

    if st.button("Ask") and question:
        if not index.records:
            st.warning("No documents indexed yet -- go to the Upload & Index tab first.")
        else:
            with st.spinner("Retrieving relevant pages..."):
                retrieved = index.search(retriever, question, top_k)

            with st.spinner("Generating answer..."):
                images = [Image.open(r.image_path).convert("RGB") for r in retrieved]
                answer = generator.answer(question, images)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved pages (used to generate the answer above)")
            cols = st.columns(len(retrieved)) if retrieved else []
            for col, record in zip(cols, retrieved):
                with col:
                    st.image(record.image_path, caption=f"{record.source} -- page {record.page_number}")
