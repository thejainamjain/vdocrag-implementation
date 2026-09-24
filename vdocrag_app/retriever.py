"""
Thin wrapper around ColQwen2.5's own encode/process calls -- no reimplemented
pooling logic (same lesson from Path A: use the reference implementation's
own methods, don't reinvent them).

Each page image produces a MULTI-vector embedding (one vector per image
patch, ~700-1000 patches per page depending on resolution), not a single
pooled vector like VDocRAG's own retriever. This is the actual retrieval
upgrade Path B buys over the original paper's approach.
"""

import logging
from typing import List

import torch
from PIL import Image

logger = logging.getLogger("vdocrag")


class Retriever:
    def __init__(self, model_manager):
        self._mm = model_manager

    @torch.inference_mode()
    def encode_query(self, query: str) -> torch.Tensor:
        model, processor = self._mm.use_retriever()
        batch = processor.process_queries([query]).to(model.device)
        embedding = model(**batch)
        return embedding[0].to("cpu")  # (num_query_tokens, dim)

    @torch.inference_mode()
    def encode_document(self, image: Image.Image) -> torch.Tensor:
        """One image at a time (see index.py for why) -- returns a single
        page's multi-vector embedding, shape (num_patches, dim)."""
        model, processor = self._mm.use_retriever()
        batch = processor.process_images([image]).to(model.device)
        embedding = model(**batch)
        return embedding[0].to("cpu")

    def encode_documents(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Sequential per-image loop, not a single batched call -- each page
        can produce a different number of patches, and processing one at a
        time avoids needing to reason about colpali_engine's internal padding/
        masking behavior for a ragged batch. Fine at demo scale (tens to a
        few hundred pages); revisit with proper batching if indexing speed
        becomes a bottleneck."""
        return [self.encode_document(img) for img in images]

    def score(self, query_embedding: torch.Tensor, doc_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """MaxSim/ColBERT-style scoring via the processor's own method --
        one score per document, for a single query. Loops per-document rather
        than passing the whole list at once, for the same "don't assume
        colpali_engine's internal batching contract" reason as encode_documents.
        Returns a 1D tensor of length len(doc_embeddings)."""
        _, processor = self._mm.use_retriever()
        scores = []
        for doc_emb in doc_embeddings:
            s = processor.score_multi_vector([query_embedding], [doc_emb])
            scores.append(s[0, 0].item())
        return torch.tensor(scores)