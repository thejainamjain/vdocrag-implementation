"""Tests that don't need a GPU or the real models -- pure logic checks on
MultiVectorIndex (save/load round-trip, remove_source, search ordering via a
fake retriever)."""

import torch

from vdocrag_app.index import MultiVectorIndex, PageRecord


class FakeRetriever:
    """Stands in for the real Retriever -- encode_query/score just compare
    raw floats so we can assert search() returns results in score order."""

    def encode_query(self, query: str):
        return torch.tensor([float(query)])

    def score(self, query_embedding, doc_embeddings):
        # "similarity" = -abs difference between query float and doc float
        return torch.tensor([-abs(query_embedding.item() - d.item()) for d in doc_embeddings])


def test_search_returns_closest_match_first():
    index = MultiVectorIndex()
    index.add_batch([
        PageRecord(doc_id="a", source="doc.pdf", page_number=0, image_path="a.png", embedding=torch.tensor([1.0])),
        PageRecord(doc_id="b", source="doc.pdf", page_number=1, image_path="b.png", embedding=torch.tensor([5.0])),
        PageRecord(doc_id="c", source="doc.pdf", page_number=2, image_path="c.png", embedding=torch.tensor([9.0])),
    ])

    results = index.search(FakeRetriever(), query="5", top_k=2)

    assert results[0].doc_id == "b"  # exact match, closest to query=5
    assert len(results) == 2


def test_remove_source_only_removes_matching_pages():
    index = MultiVectorIndex()
    index.add_batch([
        PageRecord(doc_id="x1", source="one.pdf", page_number=0, image_path="x1.png", embedding=torch.tensor([1.0])),
        PageRecord(doc_id="y1", source="two.pdf", page_number=0, image_path="y1.png", embedding=torch.tensor([2.0])),
    ])

    removed = index.remove_source("one.pdf")

    assert removed == 1
    assert len(index.records) == 1
    assert index.records[0].source == "two.pdf"


def test_save_and_load_round_trip(tmp_path):
    index = MultiVectorIndex()
    index.add(PageRecord(doc_id="a", source="doc.pdf", page_number=0, image_path="a.png", embedding=torch.tensor([1.0, 2.0])))

    index.save(str(tmp_path))
    loaded = MultiVectorIndex.load(str(tmp_path))

    assert len(loaded.records) == 1
    assert loaded.records[0].doc_id == "a"
    assert torch.equal(loaded.records[0].embedding, torch.tensor([1.0, 2.0]))
