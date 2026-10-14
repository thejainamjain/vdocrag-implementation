"""
Scoring + metrics for the single-vector vs. multi-vector retrieval ablation.

Pure NumPy on purpose: the GPU is only needed to ENCODE pages/queries once
(see retrieval_ablation.py). Everything after that -- both scoring methods,
ranking, metrics, confidence intervals -- is CPU math on the stored
embeddings, so (a) it can be unit-tested without a GPU/torch, and (b) both
methods are scored from the IDENTICAL embeddings, which is what makes this a
controlled ablation of the scoring mechanism rather than a model comparison.

Two methods:
  maxsim : ColBERT-style late interaction -- every query token finds its
           best-matching page patch, scores are summed. (The real system.)
  meanpool: mean-pool all page patches into ONE vector, mean-pool all query
           tokens into ONE vector, cosine similarity. This mimics the
           single-vector-per-page design of VDocRAG's own retriever.

CAVEAT that must travel with any number this produces: ColQwen2.5 was
trained for MaxSim, not for pooled single-vector retrieval, whereas VDocRAG's
retriever IS trained for single-vector retrieval (RCR/RCG pretraining). So
the meanpool baseline here is a naive pooled baseline and likely UNDERSTATES
what a properly trained single-vector retriever achieves. This ablation
isolates the scoring mechanism on identical embeddings; it does not measure
"VDocRAG's retriever vs. ColQwen2.5".
"""

import math
from typing import Dict, List, Sequence

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def mean_pool(embs: Sequence[np.ndarray]) -> np.ndarray:
    """List of (n_i, d) token/patch embedding arrays -> (len, d) array of
    L2-normalized single vectors (mean over tokens, then normalize)."""
    return l2_normalize(np.stack([e.astype(np.float32).mean(axis=0) for e in embs]))


def cosine_matrix(q_vecs: np.ndarray, d_vecs: np.ndarray) -> np.ndarray:
    """(Q,d) x (P,d) -> (Q,P). Inputs assumed already L2-normalized."""
    return q_vecs @ d_vecs.T


def maxsim_matrix(
    query_embs: Sequence[np.ndarray],
    doc_embs: Sequence[np.ndarray],
    doc_block: int = 32,
) -> np.ndarray:
    """ColBERT MaxSim for every (query, doc) pair.

    score(q, d) = sum over query tokens i of max over doc patches j of q_i . d_j

    Computed block-wise over docs so memory stays bounded: for each block of
    docs we form ONE big (all-query-tokens x block-patches) similarity
    matrix, take a segmented max over each doc's patches, then a segmented
    sum over each query's tokens.
    """
    q_lens = np.array([q.shape[0] for q in query_embs])
    q_cat = np.concatenate(query_embs, axis=0).astype(np.float32)
    q_starts = np.concatenate([[0], np.cumsum(q_lens)[:-1]])

    n_q, n_d = len(query_embs), len(doc_embs)
    out = np.zeros((n_q, n_d), dtype=np.float32)

    for b0 in range(0, n_d, doc_block):
        block = doc_embs[b0:b0 + doc_block]
        d_lens = np.array([d.shape[0] for d in block])
        d_cat = np.concatenate(block, axis=0).astype(np.float32)
        d_starts = np.concatenate([[0], np.cumsum(d_lens)[:-1]])

        sim = q_cat @ d_cat.T                                   # (Tq, Tm_block)
        per_doc_max = np.maximum.reduceat(sim, d_starts, axis=1)  # (Tq, B)
        per_query_sum = np.add.reduceat(per_doc_max, q_starts, axis=0)  # (Q, B)
        out[:, b0:b0 + len(block)] = per_query_sum
    return out


def best_gold_ranks(
    scores: np.ndarray,
    gold_cols: List[List[int]],
    subset_cols: np.ndarray,
) -> np.ndarray:
    """Rank (1-based) of the best-ranked gold page for each query, among ONLY
    the docs in `subset_cols` (indices into the full score matrix's columns).

    Every query's gold column(s) must be inside subset_cols. Ties are
    resolved optimistically for the gold page (rank = 1 + number of docs with
    a STRICTLY higher score); with continuous float scores exact ties are
    vanishingly rare, and this is the standard convention.
    """
    sub = scores[:, subset_cols]
    pos_of = {int(c): i for i, c in enumerate(subset_cols)}
    ranks = np.zeros(scores.shape[0], dtype=np.int64)
    for qi, gold in enumerate(gold_cols):
        gold_pos = [pos_of[g] for g in gold if g in pos_of]
        if not gold_pos:
            raise ValueError(f"query {qi}: no gold page inside the corpus subset")
        best_gold_score = sub[qi, gold_pos].max()
        ranks[qi] = 1 + int((sub[qi] > best_gold_score).sum())
    return ranks


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    m = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    m["mrr"] = float((1.0 / ranks).mean())
    m["ndcg@10"] = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]))
    return m


def per_question_values(ranks: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-question metric values, needed for paired significance testing."""
    return {
        "recall@1": (ranks <= 1).astype(np.float64),
        "recall@3": (ranks <= 3).astype(np.float64),
        "mrr": 1.0 / ranks,
    }


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> Dict[str, float]:
    """Paired bootstrap over QUESTIONS for mean(a) - mean(b).

    Paired (same questions scored by both methods) is the right test here,
    since question difficulty is shared between methods; an unpaired
    comparison would throw that information away and widen the interval.
    Returns the observed mean difference and a 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
    }
