# VDocRAG demo -- Path B

Retrieval: **ColQwen2.5-3B** (`vidore/colqwen2.5-v0.2`, multi-vector / ColBERT-style).
Generation: **Qwen2.5-VL-7B-Instruct**. Both native `transformers`/`colpali_engine`
support -- no `trust_remote_code`, proper KV-cache generation.

Both 4-bit quantized. Fallback pair (`ColQwen2-2B` + `Qwen2-VL-2B-Instruct`) is
built in -- toggle "Use smaller fallback models" in the app's sidebar if the
bigger pair doesn't fit your GPU.

## Run

Locally (needs a CUDA GPU):
```bash
pip install -r requirements-colab.txt
streamlit run app.py
```

In Colab: open `notebooks/run_in_colab.ipynb`, set `REPO_URL` to your repo,
run all cells. Gives a public `*.loca.lt` link (see notebook for the one-time
tunnel-password step).

## Structure
```
vdocrag_app/
  config.py          model IDs, quantization config, TOP_K
  model_manager.py    loads both models once, 4-bit
  retriever.py         ColQwen2.5 encode_query / encode_document / score
  generator.py          Qwen2.5-VL chat-template generation
  index.py               multi-vector store (brute-force MaxSim search), Drive-persisted
  ingest.py                PDF -> page images
app.py                       Streamlit UI (Upload & Index / Ask tabs)
notebooks/run_in_colab.ipynb  the only file you open directly in Colab
tests/test_index.py            basic tests, no GPU needed
```

## Known open items
- VRAM footprint of the bigger pair (~7-8GB weights estimated) not yet
  confirmed on real hardware -- toggle the fallback checkbox if it OOMs.
- `Retriever.encode_documents` / `.score` loop one image/doc at a time rather
  than batching -- fine at demo scale, revisit if indexing many pages gets slow.
- License note: `vidore/colqwen2.5-v0.2` is MIT/Apache-2.0. If you ever swap in
  the community `ColQwen2-7B` (T-Systems) instead, note it's CC-BY-NC-4.0
  (non-commercial) -- fine for coursework/research, not for commercial use.
