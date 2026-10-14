import math
import numpy as np
from eval.ablation_scoring import (
    l2_normalize, mean_pool, cosine_matrix, maxsim_matrix,
    best_gold_ranks, metrics_from_ranks, per_question_values, paired_bootstrap_diff,
)


def _rand_unit(rng, n, d):
    return l2_normalize(rng.normal(size=(n, d)).astype(np.float32))


def _naive_maxsim(q, d):
    return float((q @ d.T).max(axis=1).sum())


def test_maxsim_matches_naive_reference_on_ragged_shapes():
    rng = np.random.default_rng(0)
    qs = [_rand_unit(rng, n, 16) for n in (3, 7, 5, 4)]            # ragged query lengths
    ds = [_rand_unit(rng, n, 16) for n in (11, 2, 30, 8, 19, 1, 6)]  # ragged doc lengths, incl. len 1
    for block in (1, 2, 3, 32):                                       # incl. block not dividing n_docs
        got = maxsim_matrix(qs, ds, doc_block=block)
        assert got.shape == (4, 7)
        for i, q in enumerate(qs):
            for j, d in enumerate(ds):
                assert abs(got[i, j] - _naive_maxsim(q, d)) < 1e-4, (block, i, j)


def test_maxsim_finds_small_needle_that_meanpool_dilutes():
    # Query = one distinctive token. Target doc has that token's twin hidden in
    # ONE patch among 200 noise patches; distractor docs are pure noise but
    # their MEAN happens to be closer to the query than the target's diluted mean.
    rng = np.random.default_rng(1)
    d = 32
    needle = _rand_unit(rng, 1, d)
    query = [needle.copy()]
    noise_dir = _rand_unit(rng, 1, d)[0]
    # distractors: all patches lean toward a shared direction (mean-pool keeps it)
    def lean(n):
        return l2_normalize(0.35 * noise_dir + _rand_unit(rng, n, d))
    target = lean(200); target[57] = needle[0]
    docs = [lean(200) for _ in range(30)] + [target]
    gold = [[30]]

    ms = maxsim_matrix(query, docs)
    mp = cosine_matrix(mean_pool(query), mean_pool(docs))
    r_ms = best_gold_ranks(ms, gold, np.arange(31))[0]
    r_mp = best_gold_ranks(mp, gold, np.arange(31))[0]
    assert r_ms == 1, r_ms           # exact patch match -> score 1.0, unbeatable
    assert r_mp > r_ms, (r_ms, r_mp)  # pooled vector washes the needle out


def test_best_gold_ranks_subset_and_multi_gold():
    scores = np.array([[0.1, 0.9, 0.5, 0.7, 0.3]])
    # all docs: gold={2} has 2 docs strictly better (0.9, 0.7) -> rank 3
    assert best_gold_ranks(scores, [[2]], np.arange(5))[0] == 3
    # multi-gold: best of {2,3} is doc 3 (0.7), only 0.9 beats it -> rank 2
    assert best_gold_ranks(scores, [[2, 3]], np.arange(5))[0] == 2
    # subset removes the 0.9 doc -> gold {2} now rank 2
    assert best_gold_ranks(scores, [[2]], np.array([0, 2, 3, 4]))[0] == 2


def test_metrics_and_values():
    ranks = np.array([1, 2, 4, 11])
    m = metrics_from_ranks(ranks)
    assert m["recall@1"] == 0.25 and m["recall@3"] == 0.5 and m["recall@10"] == 0.75
    assert abs(m["mrr"] - (1 + 0.5 + 0.25 + 1 / 11) / 4) < 1e-9
    exp_ndcg = (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(5) + 0) / 4
    assert abs(m["ndcg@10"] - exp_ndcg) < 1e-9
    assert per_question_values(ranks)["recall@1"].tolist() == [1, 0, 0, 0]


def test_paired_bootstrap_detects_real_gap_and_null():
    rng = np.random.default_rng(2)
    a = (rng.random(150) < 0.9).astype(float)
    b = (rng.random(150) < 0.6).astype(float)
    r = paired_bootstrap_diff(a, b)
    assert r["ci95_low"] > 0 and r["mean_diff"] > 0.15      # clear gap: CI excludes 0
    same = paired_bootstrap_diff(a, a)
    assert same["mean_diff"] == 0 and same["ci95_low"] == 0 and same["ci95_high"] == 0
