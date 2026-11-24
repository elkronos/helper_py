"""Tests for appropriateness_evaluator. A deterministic char-trigram embedder stands in for
sentence-transformers so the suite runs offline in a second or two: `pytest -q`."""

import pickle
import zlib

import numpy as np
import pandas as pd
import pytest

import appropriateness_evaluator as A


def trigram_embedder(texts, dim=256):
    out = np.zeros((len(texts), dim), np.float32)
    for r, t in enumerate(texts):
        s = f"  {t}  "
        for j in range(len(s) - 2):
            h = zlib.crc32(s[j:j + 3].encode())
            out[r, h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    return out


def no_embedder(texts):
    raise AssertionError("embedder should not have been called")


def build(rows, **kw):
    return A.ReferenceIndex.from_rows(rows, embedder=kw.pop("embedder", trigram_embedder), **kw)


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    def refuse(name):
        raise RuntimeError("sentence-transformers is unavailable in tests")
    monkeypatch.setattr(A, "_sentence_transformer", refuse)


# ---- normalization ------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("2 eggs", "eggs"),
    ("1.5 cups flour", "flour"),
    ("1 1/2 cups of flour", "flour"),
    ("½ cup sugar", "sugar"),
    ("1½ cups milk", "milk"),
    ("500g pasta", "pasta"),
    ("2-3 cloves garlic", "garlic"),
    ("2 garlic cloves", "garlic"),
    ("flour 2 cups", "flour"),
    ("1 (14.5 oz) can diced tomatoes", "diced tomatoes"),
    ("a pinch of salt", "salt"),
    ("cloves", "cloves"),
    ("ground cloves", "ground cloves"),
    ("1 tsp ground cloves", "ground cloves"),
    ("7up", "7up"),
    ("V8 juice", "v8 juice"),
    ("1 tomato", "tomato"),
    ("vitamin b 12 tablets", "vitamin b 12 tablets"),
    ("chinese 5 spice, 1 tsp", "chinese 5 spice"),
    ("#1 sauce", "sauce"),
    ("1 cup 2% milk", "2 milk"),
    ("2 1/2 cups", ""),
    ("2 cups", ""),
    ("1/2", ""),
    ("", ""),
    (None, ""),
])
def test_normalize_item(raw, expected):
    assert A.normalize_item(raw) == expected


def test_extract_items_handles_list_literals_and_containers():
    assert A.extract_items_from_row("['1 (14 oz) can tomatoes, drained', 'salt']") == ["tomatoes", "salt"]
    assert A.extract_items_from_row('["Basil", null]') == ["basil"]
    assert A.extract_items_from_row("salt; pepper | 2 eggs") == ["salt", "pepper", "eggs"]
    assert A.extract_items_from_row(np.array(["salt", "garlic"])) == ["salt", "garlic"]
    assert A.extract_items_from_row(pd.Series(["salt", None])) == ["salt"]
    assert A.extract_items_from_row(float("nan")) == []
    assert A.extract_items_from_row(pd.NA) == []
    assert A.extract_items_from_row(None) == []


def test_counts_are_per_row_and_merge_quantities():
    idx = build(["2 eggs, 3 eggs, flour", "1 egg, eggs", None, ""])
    assert idx.n_rows == 2
    assert idx.counts[idx.index_of("eggs")] == 2


# ---- regressions from the review -----------------------------------------------------------

def test_frequent_exact_items_are_never_vetoed_as_outliers():
    rng = np.random.default_rng(0)
    vocab = [f"ingredient{w}" for w in range(120)]
    rows = [", ".join(rng.choice(vocab, 8, replace=False)) + ", salt, garlic, olive oil"
            for _ in range(300)]
    ev = A.AppropriatenessEvaluator("ctx", build(rows))
    for item in ["salt", "garlic", "olive oil"]:
        assert ev.evaluate(item).decision == A.APPROPRIATE


def test_support_floor_scales_with_dataset_size():
    rng = np.random.default_rng(0)
    vocab = [f"ingredient{w}" for w in range(1000)]
    rows = [", ".join(rng.choice(vocab, 5, replace=False)) for _ in range(5000)]
    idx = build(rows, embedder=no_embedder)
    results = A.AppropriatenessEvaluator("ctx", idx).evaluate_many(vocab)
    # ~25 occurrences each: established, even though that is only 0.5% of rows
    assert all(r.decision == A.APPROPRIATE for r in results if r.best_count >= 2)


def test_rare_lookalike_does_not_mask_established_neighbour():
    idx = build(["parmesan, basil"] * 50 + ["parmesan cheesy crisps"])
    ev = A.AppropriatenessEvaluator("ctx", idx, A.PRESETS["normal"]).evaluate("parmesan cheesy crisp")
    assert ev.decision == A.APPROPRIATE and ev.best_match == "parmesan"


def test_pickle_files_are_rejected_without_executing(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    path = tmp_path / "evil.idx"
    path.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ValueError, match="not a saved index"):
        A.ReferenceIndex.load(str(path))
    assert not marker.exists()


# ---- behaviour --------------------------------------------------------------------------------

def test_score_and_decision_agree():
    idx = build(A._DEMO_ROWS * 10 + ["rigatoni, anchovy"])
    ev = A.AppropriatenessEvaluator("ctx", idx, A.PRESETS["strict"])
    probe = idx.items + ["parmesan cheese", "cheetos", "garlic powder", "motor oil", "tofu", "basil leaves"]
    for r in ev.evaluate_many(probe):
        assert (r.score >= 0.5) == (r.decision == A.APPROPRIATE), r


def test_exact_matches_never_load_the_embedder():
    idx = build(A._DEMO_ROWS, embedder=no_embedder)
    rs = A.AppropriatenessEvaluator("ctx", idx).evaluate_many(["Garlic", "2 cloves garlic", "basil"])
    assert [r.decision for r in rs] == [A.APPROPRIATE] * 3
    assert [r.item for r in rs] == ["Garlic", "2 cloves garlic", "basil"]


def test_unseen_items_without_embedder_are_undetermined():
    idx = A.ReferenceIndex.from_rows(A._DEMO_ROWS)  # falls back to the refused real model
    rs = A.AppropriatenessEvaluator("ctx", idx).evaluate_many(["garlic", "tofu"])
    assert [r.decision for r in rs] == [A.APPROPRIATE, A.UNDETERMINED]


def test_save_load_roundtrip(tmp_path):
    idx = build(A._DEMO_ROWS)
    path = tmp_path / "pasta.idx"
    idx.save(str(path))
    assert path.exists()
    loaded = A.ReferenceIndex.load(str(path), embedder=trigram_embedder)
    assert loaded.items == idx.items and loaded.n_rows == idx.n_rows
    np.testing.assert_allclose(loaded.embeddings, idx.embeddings, atol=1e-6)
    a = A.AppropriatenessEvaluator("ctx", idx).evaluate("parmesan cheese")
    b = A.AppropriatenessEvaluator("ctx", loaded).evaluate("parmesan cheese")
    assert a.decision == b.decision and a.score == pytest.approx(b.score, abs=1e-6)


def test_embedding_dimension_mismatch_is_reported(tmp_path):
    path = tmp_path / "pasta.idx"
    build(A._DEMO_ROWS).save(str(path))
    loaded = A.ReferenceIndex.load(str(path), embedder=lambda t: trigram_embedder(t, dim=64))
    with pytest.raises(ValueError, match="64-d"):
        A.AppropriatenessEvaluator("ctx", loaded).evaluate("tofu")


def test_explain_lists_neighbours_without_self():
    idx = build(A._DEMO_ROWS)
    r = A.AppropriatenessEvaluator("ctx", idx, k_neighbors=3).evaluate("garlic", explain=True)
    assert len(r.neighbors) == 3 and all(n != "garlic" for n, _, _ in r.neighbors)


class FakeLLM:
    available = True

    def __init__(self, p):
        self.p, self.calls = p, 0

    def score(self, context, item):
        self.calls += 1
        return self.p, "not pasta"


def test_llm_veto_is_cached_and_binding():
    llm = FakeLLM(0.1)
    idx = build(A._DEMO_ROWS, embedder=no_embedder)
    t = A.replace(A.PRESETS["normal"], llm_enabled=True)
    ev = A.AppropriatenessEvaluator("pasta", idx, t, llm_veto=llm)
    rs = ev.evaluate_many(["garlic", "Garlic", "1 clove garlic"])
    assert llm.calls == 1
    assert all(r.decision == A.INAPPROPRIATE and r.reason.startswith("LLM veto") for r in rs)
    assert all(r.score < 0.5 for r in rs)


def test_llm_reply_parsing():
    assert A.LLMVeto._parse('```json\n{"p": 0.8, "reason": "ok"}\n```') == (0.8, "ok")
    assert A.LLMVeto._parse("Step 1: I think 0 issues") is None
    assert A.LLMVeto._parse('{"p": 85}') is None
    assert A.LLMVeto._parse('{"p": "NaN"}') is None
    assert A.LLMVeto._parse("[1, 2]") is None


def test_thresholds_validate():
    with pytest.raises(ValueError):
        A.Thresholds(min_sim=1.5)
    with pytest.raises(ValueError):
        A.Thresholds(min_count=0)
    assert A.Thresholds(min_freq=0.04, min_count=2).support_floor(10_000) == 2 / 10_000
    assert A.Thresholds(min_freq=0.04, min_count=None).support_floor(10_000) == 0.04


# ---- CLI --------------------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(A, "_sentence_transformer", lambda name: trigram_embedder)
    csv = tmp_path / "recipes.csv"
    pd.DataFrame({"ingredients": [repr(r.split(", ")) for r in A._DEMO_ROWS * 4]}).to_csv(csv, index=False)
    idx_path = tmp_path / "pasta.idx"

    assert A.main(["-q", "-d", str(csv), "-c", "ingredients", "--save", str(idx_path),
                   "--context", "pasta", "garlic", "motor oil"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("garlic: Appropriate")
    assert lines[1].startswith("motor oil: Inappropriate")

    assert A.main(["-q", "-m", str(idx_path), "--json", "--context", "pasta", "-i", "basil, 2 cups"]) == 0
    import json
    outs = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [o["decision"] for o in outs] == [A.APPROPRIATE, A.INAPPROPRIATE]

    assert A.main(["-q", "-m", str(tmp_path / "missing.idx"), "--context", "x", "y"]) == 2
