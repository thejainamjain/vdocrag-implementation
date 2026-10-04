"""
Runs a small OpenDocVQA subset (default: ChartQA config) through the existing
Path B pipeline and reports Recall@k, nDCG@k, EM, F1, and relaxed accuracy.

CONFIRMED SCHEMA (checked directly against the real dataset, not guessed):
    QA fields:     query_id, query, answers (list), relevant_doc_ids (list),
                   dataset_names (list)
    Corpus fields: doc_id, image, dataset_name
`doc_id` values (e.g. "chartqa/166.png") match `relevant_doc_ids` entries
directly -- no ID translation needed.

TWO DELIBERATE DEVIATIONS from "just query the raw dataset", both load-bearing:

1. CORPUS REDUCTION. The real corpus is large (20,882 pages for ChartQA
   alone) -- encoding all of it is a multi-hour job, not a quick Colab-session
   benchmark. Instead we build a REDUCED corpus per run: every gold page for
   the sampled questions, plus a batch of random distractor pages, totalling
   --corpus_size (default 400). This is still a genuine open-domain retrieval
   test (the model must find the right page among real distractors it has
   never seen before) -- it is NOT the full-corpus number VDocRAG's own paper
   reports, and Section V-B/VII of the paper must say so explicitly.

2. INSTRUCTION-STRIPPING. The dataset's `query` field bundles NTT's own
   retrieval instruction template around the actual question
   ("Instruct: ...\nQuery: <question>"). ColQwen2.5 was not trained with that
   template (ColPali-style models are trained on raw questions), and the app
   itself queries with raw question text -- so for a fair, consistent-with-
   the-app test, we extract just the text after "\nQuery: " before passing it
   to the retriever.

Usage (run in Colab, after `login(token=...)` for the gated corpus dataset):
    python -m eval.run_opendocvqa_eval --max_examples 30 --corpus_size 400 --top_k 3
"""

import argparse
import json
import logging
import os
import random
import time

from datasets import load_dataset

from eval.metrics import recall_at_k, ndcg_at_k, aggregate_qa_metrics
from vdocrag_app.generator import Generator
from vdocrag_app.index import MultiVectorIndex, PageRecord
from vdocrag_app.model_manager import ModelManager
from vdocrag_app.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vdocrag")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="chartqa", help="OpenDocVQA sub-dataset config")
    p.add_argument("--max_examples", type=int, default=30,
                    help="Number of QUESTIONS to evaluate -- keep small for a single Colab session")
    p.add_argument("--corpus_size", type=int, default=400,
                    help="Reduced-corpus size (gold pages + random distractors) -- NOT the full 20k+ corpus")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_fallback_models", action="store_true")
    p.add_argument("--output_path", default="eval/results.json")
    return p.parse_args()


def extract_question(raw_query: str) -> str:
    """Strips NTT's 'Instruct: ...\\nQuery: ' wrapper, falls back to the raw
    string unchanged if that marker isn't present (defensive -- other
    OpenDocVQA configs may not use exactly this wrapper)."""
    marker = "\nQuery: "
    if marker in raw_query:
        return raw_query.split(marker, 1)[1].strip()
    return raw_query.strip()


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.info(f"Loading OpenDocVQA[{args.config}] QA pairs...")
    qa_ds = load_dataset("NTT-hil-insight/OpenDocVQA", args.config, split="test")
    qa_ds = qa_ds.select(range(min(args.max_examples, len(qa_ds))))

    logger.info(f"Loading OpenDocVQA-Corpus[{args.config}] (full index, then reducing)...")
    full_corpus = load_dataset("NTT-hil-insight/OpenDocVQA-Corpus", args.config, split="test")
    doc_id_to_idx = {d: i for i, d in enumerate(full_corpus["doc_id"])}

    # --- build the reduced corpus: every gold page + random distractors ---
    gold_doc_ids = set()
    for ex in qa_ds:
        gold_doc_ids.update(ex["relevant_doc_ids"])

    all_doc_ids = full_corpus["doc_id"]
    distractor_pool = [d for d in all_doc_ids if d not in gold_doc_ids]
    n_distractors = max(0, args.corpus_size - len(gold_doc_ids))
    distractors = random.sample(distractor_pool, min(n_distractors, len(distractor_pool)))
    reduced_doc_ids = list(gold_doc_ids) + distractors
    logger.info(f"Reduced corpus: {len(gold_doc_ids)} gold pages + {len(distractors)} distractors "
                 f"= {len(reduced_doc_ids)} total (full corpus was {len(all_doc_ids)})")

    mm = ModelManager(use_fallback=args.use_fallback_models)
    mm.load()
    retriever = Retriever(mm)
    generator = Generator(mm)

    logger.info(f"Indexing {len(reduced_doc_ids)} corpus pages...")
    index = MultiVectorIndex()
    t0 = time.time()
    for doc_id in reduced_doc_ids:
        row = full_corpus[doc_id_to_idx[doc_id]]
        emb = retriever.encode_document(row["image"])
        index.add(PageRecord(doc_id=doc_id, source=args.config, page_number=doc_id_to_idx[doc_id],
                               image_path="", embedding=emb))
    logger.info(f"Indexed {len(index.records)} pages in {time.time() - t0:.0f}s")

    rankings, gold_pages, gold_answer, predictions = {}, {}, {}, {}
    per_question_log = []

    for i, ex in enumerate(qa_ds):
        qid = ex["query_id"]
        question = extract_question(ex["query"])
        answer = ex["answers"][0] if ex["answers"] else ""

        results = index.search(retriever, question, args.top_k)
        rankings[qid] = [r.doc_id for r in results]
        gold_pages[qid] = ex["relevant_doc_ids"]
        gold_answer[qid] = answer

        # Generate using the ACTUAL retrieved images (not necessarily the
        # gold page) -- measures the end-to-end pipeline including retrieval
        # mistakes, which is the honest number to report.
        images = [full_corpus[doc_id_to_idx[r.doc_id]]["image"] for r in results]
        pred = generator.answer(question, images)
        predictions[qid] = pred

        correct_retrieved = bool(set(rankings[qid]) & set(gold_pages[qid]))
        per_question_log.append({
            "qid": qid, "question": question, "gold_answer": answer,
            "predicted_answer": pred, "correct_page_retrieved": correct_retrieved,
            "retrieved_pages": rankings[qid], "gold_pages": gold_pages[qid],
        })
        logger.info(f"[{i+1}/{len(qa_ds)}] gold={answer!r} pred={pred!r} "
                     f"correct_page_retrieved={correct_retrieved}")

    retrieval_metrics = {
        f"recall@{args.top_k}": recall_at_k(rankings, gold_pages, args.top_k),
        f"ndcg@{args.top_k}": ndcg_at_k(rankings, gold_pages, args.top_k),
    }
    qa_metrics = aggregate_qa_metrics(predictions, gold_answer)

    print(f"\n=== Reduced-corpus setup: {len(reduced_doc_ids)} pages "
          f"(NOT the full {len(all_doc_ids)}-page corpus -- see script docstring) ===")
    print("=== Retrieval metrics ===")
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v:.3f}")
    print("=== QA metrics ===")
    for k, v in qa_metrics.items():
        print(f"  {k}: {v}")

    # Surface clean failure-case candidates for the paper: wrong answer, but
    # the correct page WAS retrieved -- means the error is in generation,
    # not retrieval, which is the cleaner story to write up.
    wrong = [q for q in per_question_log
             if q["predicted_answer"].strip().lower() != q["gold_answer"].strip().lower()
             and q["correct_page_retrieved"]]
    print(f"\n=== {len(wrong)} wrong answers WITH correct page retrieved (good failure-case candidates) ===")
    for q in wrong[:5]:
        print(f"  Q: {q['question']}\n  Gold: {q['gold_answer']!r}  Pred: {q['predicted_answer']!r}\n")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config, "n_examples": len(qa_ds), "top_k": args.top_k,
            "reduced_corpus_size": len(reduced_doc_ids), "full_corpus_size": len(all_doc_ids),
            "retrieval_metrics": retrieval_metrics, "qa_metrics": qa_metrics,
            "per_question": per_question_log,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results (including per-question log) saved to {args.output_path}")


if __name__ == "__main__":
    main()