"""
Multi-vector index: stores one variable-length embedding tensor per page
(from Retriever.encode_document), plus metadata (source pdf, page number,
page image path). Search scores a query against every stored doc via
Retriever.score() -- exact, brute-force, no ANN structure.

This is intentionally simple (a Python list, not FAISS): ColBERT-style
multi-vector scores aren't plain cosine similarity, so a standard
vector-index library like FAISS doesn't directly apply the way it did for
VDocRAG's single-vector embeddings. At demo scale (tens-hundreds of pages)
brute-force scoring the whole corpus per query is fast enough. Revisit with
something like PLAID (colpali_engine's own experimental fast-search support)
only if corpus size becomes the bottleneck.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger("vdocrag")


@dataclass
class PageRecord:
    doc_id: str          # e.g. "annual_report.pdf::page_3"
    source: str           # original filename
    page_number: int
    image_path: str        # where the page image is saved on disk
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)


class MultiVectorIndex:
    def __init__(self):
        self.records: List[PageRecord] = []

    def add(self, record: PageRecord):
        self.records.append(record)

    def add_batch(self, records: List[PageRecord]):
        self.records.extend(records)

    def remove_source(self, source: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.source != source]
        removed = before - len(self.records)
        if removed:
            logger.info(f"Removed {removed} pages for source '{source}'")
        return removed

    def search(self, retriever, query: str, top_k: int) -> List[PageRecord]:
        if not self.records:
            return []
        query_embedding = retriever.encode_query(query)
        doc_embeddings = [r.embedding for r in self.records]
        scores = retriever.score(query_embedding, doc_embeddings)
        top_k = min(top_k, len(self.records))
        top_idx = torch.topk(scores, top_k).indices.tolist()
        return [self.records[i] for i in top_idx]

    # --- persistence -------------------------------------------------------
    # Embeddings saved via torch.save (list of tensors, ragged shapes are
    # fine); metadata saved as plain JSON alongside for readability.

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        torch.save([r.embedding for r in self.records], os.path.join(dir_path, "embeddings.pt"))
        meta = [
            {"doc_id": r.doc_id, "source": r.source, "page_number": r.page_number, "image_path": r.image_path}
            for r in self.records
        ]
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Index saved to {dir_path} ({len(self.records)} pages)")

    @classmethod
    def load(cls, dir_path: str) -> "MultiVectorIndex":
        idx = cls()
        embeddings_path = os.path.join(dir_path, "embeddings.pt")
        meta_path = os.path.join(dir_path, "meta.json")
        if not (os.path.exists(embeddings_path) and os.path.exists(meta_path)):
            logger.info(f"No existing index found at {dir_path}, starting empty.")
            return idx

        embeddings = torch.load(embeddings_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        for m, emb in zip(meta, embeddings):
            idx.records.append(PageRecord(embedding=emb, **m))
        logger.info(f"Loaded index from {dir_path} ({len(idx.records)} pages)")
        return idx
