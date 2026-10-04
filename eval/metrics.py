"""
Retrieval and QA metrics for the OpenDocVQA benchmark subset.

No model code here -- pure functions over plain Python data, so these are
testable without a GPU (see tests/test_metrics.py if you add one, following
the same no-GPU-needed pattern as tests/test_index.py).
"""

import math
import re
import string
from typing import Dict, List


# ---------- Retrieval metrics ----------
# NOTE: OpenDocVQA's `relevant_doc_ids` is a LIST per question (occasionally
# more than one correct page) -- gold_pages below takes a set/list of
# acceptable IDs per question, not a single string, matching that schema.

def recall_at_k(rankings: Dict[str, List[str]], gold_pages: Dict[str, List[str]], k: int) -> float:
    """rankings: {question_id: [page_id, page_id, ...]} ordered best-first.
    gold_pages: {question_id: [correct_page_id, ...]}."""
    hits = 0
    for qid, correct_set in gold_pages.items():
        retrieved_top_k = set(rankings.get(qid, [])[:k])
        if retrieved_top_k & set(correct_set):
            hits += 1
    return hits / len(gold_pages) if gold_pages else 0.0


def ndcg_at_k(rankings: Dict[str, List[str]], gold_pages: Dict[str, List[str]], k: int) -> float:
    scores = []
    for qid, correct_set in gold_pages.items():
        correct_set = set(correct_set)
        retrieved = rankings.get(qid, [])[:k]
        dcg = 0.0
        for i, page in enumerate(retrieved):
            if page in correct_set:
                dcg = 1.0 / math.log2(i + 2)  # first correct hit only, standard for single-relevant-ish cases
                break
        idcg = 1.0  # best possible: a correct page at rank 1 -> 1/log2(2) = 1.0
        scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0


# ---------- QA metrics ----------

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def relaxed_accuracy(pred: str, gold: str, tolerance: float = 0.05) -> float:
    """ChartQA-style: numeric answers within `tolerance` relative error count as correct."""
    def try_float(x):
        try:
            return float(x.strip().rstrip("%").replace(",", ""))
        except ValueError:
            return None

    p_num, g_num = try_float(pred), try_float(gold)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return float(p_num == 0)
        return float(abs(p_num - g_num) / abs(g_num) <= tolerance)
    return exact_match(pred, gold)


def aggregate_qa_metrics(predictions: Dict[str, str], gold_answers: Dict[str, str]) -> Dict[str, float]:
    em, f1, acc = [], [], []
    for qid, gold in gold_answers.items():
        pred = predictions.get(qid, "")
        em.append(exact_match(pred, gold))
        f1.append(token_f1(pred, gold))
        acc.append(relaxed_accuracy(pred, gold))
    n = len(gold_answers)
    return {
        "exact_match": sum(em) / n if n else 0.0,
        "f1": sum(f1) / n if n else 0.0,
        "relaxed_accuracy": sum(acc) / n if n else 0.0,
        "n_questions": n,
    }