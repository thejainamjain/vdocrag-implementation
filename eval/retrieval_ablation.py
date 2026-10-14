"""
EXTENSION EXPERIMENT: single-vector vs. multi-vector retrieval, across
corpus sizes -- retrieval only (no generation), so it is cheap enough to run
on ALL 150 ChartQA test questions.

WHY THIS DESIGN (read before changing anything)
-----------------------------------------------
1. The earlier 20-question pilot hit Recall@3 = 1.000 -- a ceiling effect.
   Two methods that both score 1.0 can't be told apart, so this experiment is
   deliberately harder: every ChartQA test question (150), corpora up to
   --pool_size pages, and stricter metrics (Recall@1, MRR) alongside Recall@3/5/10.

2. ENCODE ONCE, SCORE MANY WAYS. Every page in the pool is encoded a single
   time (the only GPU-heavy step; resumable via --cache_dir). Both scoring
   methods and every corpus size are then computed from those same stored
   embeddings on CPU. Corpora are NESTED: gold pages first, then distractors
   in a fixed seeded order, so the size-N corpus is the first N pool entries.
   This makes (a) the methods directly comparable on identical embeddings and
   (b) the corpus-scaling curve nearly free.

3. THE BASELINE'S LIMITATION (state this in the paper): ColQwen2.5 is trained
   for MaxSim. The "meanpool" baseline mean-pools its patch embeddings into
   one vector -- a naive pooled baseline, NOT a retriever trained for
   single-vector retrieval like VDocRAG's. The gap measured here therefore
   isolates the scoring mechanism and likely overstates what a properly
   trained single-vector retriever would lose.

BACKBONE COMPARISON: run the same command with --use_fallback_models
(ColQwen2-2B) and a different --output_path / --cache_dir.

Usage (Colab, after HF login; run SMALL FIRST to measure seconds/page):
    python -m eval.retrieval_ablation --max_questions 30 --pool_size 300 --corpus_sizes 200 300
    python -m eval.retrieval_ablation --max_questions 150 --pool_size 2000 \
        --corpus_sizes 300 500 1000 2000 --cache_dir /content/drive/MyDrive/vdocrag_cache/colqwen25
"""

import argparse
import json
import logging
import os
import random
import re
import time

import numpy as np
from datasets import load_dataset

from eval.ablation_scoring import (
    best_gold_ranks, cosine_matrix, maxsim_matrix, mean_pool,
    metrics_from_ranks, paired_bootstrap_diff, per_question_values,
)
from vdocrag_app import config
from vdocrag_app.model_manager import ModelManager, _bnb_config
from vdocrag_app.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("vdocrag")


class RetrieverOnlyManager(ModelManager):
    """ModelManager that loads ONLY the retriever -- this experiment never
    generates, so loading the 7B generator would just waste time and VRAM.
    Mirrors ModelManager.load()'s retriever branch (same class pairing per
    use_fallback) without touching model_manager.py itself."""

    def load(self):
        if self._loaded:
            return self
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU visible -- switch Colab to a GPU runtime and restart.")

        retriever_id = config.FALLBACK_RETRIEVER_MODEL_ID if self.use_fallback else config.RETRIEVER_MODEL_ID
        compute_dtype = getattr(torch, config.QuantConfig().bnb_4bit_compute_dtype_name)
        if self.use_fallback:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            cls, proc_cls = ColQwen2, ColQwen2Processor
        else:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            cls, proc_cls = ColQwen2_5, ColQwen2_5_Processor

        logger.info(f"Loading retriever only: {retriever_id}")
        self.retriever_model = cls.from_pretrained(
            retriever_id, quantization_config=_bnb_config(), torch_dtype=compute_dtype,
            attn_implementation=config.ATTN_IMPLEMENTATION, device_map="cuda:0",
        ).eval()
        self.retriever_processor = proc_cls.from_pretrained(retriever_id)
        self.retriever_id = retriever_id
        self._loaded = True
        return self


def extract_question(raw_query: str) -> str:
    """Strip NTT's 'Instruct: ...\\nQuery: ' wrapper (see run_opendocvqa_eval.py)."""
    marker = "\nQuery: "
    return raw_query.split(marker, 1)[1].strip() if marker in raw_query else raw_query.strip()


def cache_path(cache_dir: str, doc_id: str) -> str:
    return os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id) + ".npy")


def ensure_hf_token():
    """OpenDocVQA-Corpus is gated, so an unauthenticated run dies with a
    confusing 'gated dataset' error AFTER downloading other things. Fail fast
    with the actual fix instead. NOTE: Colab's userdata.get() only works in the
    notebook kernel, not in a `!python -m ...` subprocess -- so the token must
    be exported to the environment from a notebook cell BEFORE launching."""
    from huggingface_hub import get_token
    if not (os.environ.get("HF_TOKEN") or get_token()):
        raise SystemExit(
            "No Hugging Face token visible to this process (needed for the gated "
            "OpenDocVQA-Corpus). In a notebook cell, run first:\n"
            "    import os; from google.colab import userdata\n"
            "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
            "then re-run this command in the SAME runtime (a restarted runtime loses it)."
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="chartqa")
    p.add_argument("--max_questions", type=int, default=150)
    p.add_argument("--pool_size", type=int, default=2000,
                    help="Total pages to encode (gold pages + distractors). Largest corpus size evaluated.")
    p.add_argument("--corpus_sizes", type=int, nargs="+", default=[300, 500, 1000, 2000])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_fallback_models", action="store_true",
                    help="ColQwen2-2B instead of ColQwen2.5-3B (backbone-size comparison)")
    p.add_argument("--cache_dir", default=None,
                    help="Per-page embedding cache (resumable). Default: /content/ablation_cache/<model>")
    p.add_argument("--output_path", default="eval/ablation_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_hf_token()

    qa = load_dataset("NTT-hil-insight/OpenDocVQA", args.config, split="test")
    qa = qa.select(range(min(args.max_questions, len(qa))))
    corpus = load_dataset("NTT-hil-insight/OpenDocVQA-Corpus", args.config, split="test")
    doc_id_to_idx = {d: i for i, d in enumerate(corpus["doc_id"])}

    # ---- nested pool: gold pages first, then seeded-shuffled distractors ----
    gold_ids = []
    for ex in qa:
        for d in ex["relevant_doc_ids"]:
            if d not in gold_ids:
                gold_ids.append(d)
    gold_set = set(gold_ids)
    distractors = [d for d in corpus["doc_id"] if d not in gold_set]
    random.Random(args.seed).shuffle(distractors)
    pool_ids = (gold_ids + distractors)[: max(args.pool_size, len(gold_ids))]
    pool_pos = {d: i for i, d in enumerate(pool_ids)}
    logger.info(f"{len(qa)} questions | {len(gold_ids)} unique gold pages | pool = {len(pool_ids)} pages")

    sizes = sorted({s for s in args.corpus_sizes if s <= len(pool_ids)})
    too_small = [s for s in args.corpus_sizes if s < len(gold_ids)]
    if too_small:
        raise SystemExit(f"corpus sizes {too_small} are smaller than the {len(gold_ids)} gold pages "
                         "-- every gold page must fit in every corpus. Raise those sizes.")

    # ---- load retriever, encode pool once (resumable) ----
    mm = RetrieverOnlyManager(use_fallback=args.use_fallback_models)
    mm.load()
    retriever = Retriever(mm)
    cache_dir = args.cache_dir or os.path.join(
        "/content/ablation_cache", mm.retriever_id.replace("/", "__"))
    os.makedirs(cache_dir, exist_ok=True)

    t0, encoded_now = time.time(), 0
    for i, doc_id in enumerate(pool_ids):
        path = cache_path(cache_dir, doc_id)
        if os.path.exists(path):
            continue
        emb = retriever.encode_document(corpus[doc_id_to_idx[doc_id]]["image"])
        np.save(path, emb.detach().float().cpu().numpy().astype(np.float16))
        encoded_now += 1
        if encoded_now % 25 == 0:
            rate = (time.time() - t0) / encoded_now
            logger.info(f"encoded {i + 1}/{len(pool_ids)} (new this run: {encoded_now}, "
                        f"{rate:.1f}s/page, ~{rate * (len(pool_ids) - i - 1) / 60:.0f} min left)")
    logger.info(f"Pool ready ({encoded_now} newly encoded, rest from cache) in {(time.time() - t0) / 60:.1f} min")

    doc_embs = [np.load(cache_path(cache_dir, d)) for d in pool_ids]
    query_embs = []
    for ex in qa:
        q = retriever.encode_query(extract_question(ex["query"]))
        query_embs.append(q.detach().float().cpu().numpy())
    gold_cols = [[pool_pos[d] for d in ex["relevant_doc_ids"]] for ex in qa]

    # ---- score once per method over the whole pool ----
    t1 = time.time()
    scores = {
        "maxsim": maxsim_matrix(query_embs, doc_embs),
        "meanpool": cosine_matrix(mean_pool(query_embs), mean_pool(doc_embs)),
    }
    logger.info(f"Scoring done in {time.time() - t1:.0f}s")

    # ---- evaluate each (corpus size x method) from the shared score matrices ----
    results = {"retriever": mm.retriever_id, "n_questions": len(qa), "n_gold_pages": len(gold_ids),
               "pool_size": len(pool_ids), "seed": args.seed, "by_corpus_size": {}}
    header = f"{'corpus':>7} {'method':<9} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for n in sizes:
        subset = np.arange(n)  # nested: first n pool entries (gold first)
        per_method, pq = {}, {}
        for name, S in scores.items():
            ranks = best_gold_ranks(S, gold_cols, subset)
            per_method[name] = {"metrics": metrics_from_ranks(ranks), "ranks": ranks.tolist()}
            pq[name] = per_question_values(ranks)
            m = per_method[name]["metrics"]
            print(f"{n:>7} {name:<9} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} {m['recall@5']:>6.3f} "
                  f"{m['recall@10']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>8.3f}")
        sig = {k: paired_bootstrap_diff(pq["maxsim"][k], pq["meanpool"][k]) for k in ("recall@1", "mrr")}
        for k, v in sig.items():
            print(f"{'':>7} maxsim - meanpool, {k}: {v['mean_diff']:+.3f}  "
                  f"95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}]")
        results["by_corpus_size"][str(n)] = {"methods": per_method, "paired_diff_maxsim_minus_meanpool": sig}

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()