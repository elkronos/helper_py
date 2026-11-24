import numpy as np
import pandas as pd
import pytest

import deduplicate_dataframe as m
from deduplicate_dataframe import DEFAULT_IGNORE, deduplicate_dataframe as dd


def ids(out, col="g"):
    return out[col].tolist()


# --------------------------------------------------------------------------
# Regressions for bugs found in review
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pick", ["max:score", "min:score", "random", lambda g, k: g.head(1), "first"])
def test_missing_key_rows_survive_every_pick(pick):
    # Was: every pick except first/last silently dropped all rows whose key was NaN.
    df = pd.DataFrame({"email": ["a", None, None, "a"], "score": [1, 2, 3, 4]})
    out = dd(df, on="email", pick=pick, random_state=0, log_stats=False)
    assert len(out) == 2 and out["email"].isna().sum() == 1


def test_custom_pick_receives_real_key_even_with_column_named_name():
    # Was: key was always None, or the "name" *column* when one existed.
    df = pd.DataFrame({"k": [1, 1, 2], "name": ["x", "y", "z"]})
    seen = []
    dd(df, on="k", pick=lambda g, key: (seen.append(key), g.head(1))[1], log_stats=False)
    assert seen == [1, 2]


def test_custom_pick_key_is_tuple_for_multi_column_and_entity_id_for_collapse():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"], "e": ["p", "p"]})
    seen = []
    pick = lambda g, key: (seen.append(key), g.head(1))[1]  # noqa: E731
    dd(df, on="a b", pick=pick, log_stats=False)
    dd(df, entity_by="e", pick=pick, log_stats=False)
    assert seen == [(1, "x"), 0]


def test_custom_pick_may_return_series_row_mask_or_labels():
    df = pd.DataFrame({"k": [1, 1, 2, 2], "v": [1, 2, 3, 4]}, index=list("abcd"))
    for fn in (lambda g, _: g.loc[g.v.idxmax()],      # one row as Series
               lambda g, _: g.v == g.v.max(),         # boolean mask
               lambda g, _: [g.v.idxmax()]):          # labels
        assert dd(df, on="k", pick=fn, log_stats=False).index.tolist() == ["b", "d"]


def test_custom_pick_rejects_foreign_rows():
    df = pd.DataFrame({"k": [1, 1, 2]})
    with pytest.raises(ValueError, match="not in its group"):
        dd(df, on="k", pick=lambda g, _: df.iloc[[2]], log_stats=False)


def test_custom_pick_keeps_original_values_and_works_with_duplicate_index():
    df = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]}, index=[7, 7, 7])
    out = dd(df, on="k", pick=lambda g, _: g.assign(v=-1).tail(1), log_stats=False)
    assert out["v"].tolist() == [2, 3] and out.index.tolist() == [7, 7]


def test_numpy_array_cells_are_flattened():
    # Parquet list columns arrive as numpy arrays; they were str()-ed into one token.
    df = pd.DataFrame({"id": [np.array(["a", "b"]), np.array(["b"]), np.array(["c"])]})
    assert ids(dd(df, entity_by="id", action="flag", group_id_column="g", log_stats=False)) == [0, 0, 1]


def test_stringified_nan_does_not_link():
    df = pd.DataFrame({"email": ["nan", "NaN", "<NA>", "None", "x"]})  # df.astype(str) artefacts
    assert ids(dd(df, entity_by="email", action="flag", group_id_column="g", log_stats=False)) == [0, 1, 2, 3, 4]


def test_auto_entity_ignores_foreign_keys():
    df = pd.DataFrame({"customer_id": [1, 2, 3], "product_id": [9, 9, 9],
                       "Email Address": ["a", "b", "c"], "paid": [1, 1, 1]})
    out, stats = dd(df, entity_by="auto", return_stats=True, log_stats=False)
    assert len(out) == 3 and stats["entity_by"] == ["customer_id", "Email Address"]


def test_column_named_link_is_not_hijacked_as_any_prefix():
    df = pd.DataFrame({"link": ["u1", "u1", "u2"], "email": ["a", "b", "b"]})
    _, stats = dd(df, on="link email", return_stats=True, log_stats=False)
    assert stats["on_kind"] == "exact"
    _, stats = dd(df, on="link: link email", return_stats=True, log_stats=False)
    assert stats["on_kind"] == "any"
    _, stats = dd(pd.DataFrame({"email": ["a"]}), on="any email", return_stats=True, log_stats=False)
    assert stats["on_kind"] == "any"


def test_non_string_and_spaced_column_labels():
    df = pd.DataFrame({0: [1, 1], "first name": ["x", "x"], "s": [1, 2]})
    assert len(dd(df, on=[0], log_stats=False)) == 1
    assert len(dd(df, on="first name", log_stats=False)) == 1
    assert dd(df, on=[0], pick=("max", "s"), log_stats=False)["s"].tolist() == [2]


def test_alias_conflicts_are_explicit():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="either on= or subset="):
        dd(df, on="a", subset="a", log_stats=False)
    with pytest.raises(TypeError, match="either pick= or keep="):
        dd(df, pick="first", keep="last", log_stats=False)
    with pytest.raises(TypeError, match="Unknown argument"):
        dd(df, sbuset="a", log_stats=False)


def test_numbers_link_by_value_across_int_and_float_columns():
    # int column vs float column (upcast by a NaN): 123 and 123.0 are the same id.
    df = pd.DataFrame({"b": [123, 456, 789], "c": [np.nan, 123.0, np.nan]})
    assert ids(dd(df, entity_by="b c", action="flag", group_id_column="g", log_stats=False)) == [0, 0, 1]


def test_type_tags_keep_strings_and_numbers_apart_unless_coerced():
    df = pd.DataFrame({"x": ["123", 123, True, 1]}, dtype=object)
    assert ids(dd(df, entity_by="x", action="flag", group_id_column="g", log_stats=False)) == [0, 1, 2, 3]
    out = dd(df, entity_by="x", action="flag", group_id_column="g", coerce_token_types=True, log_stats=False)
    assert ids(out) == [0, 0, 1, 2]


def test_user_ignore_values_are_case_insensitive():
    df = pd.DataFrame({"e": ["TBD", "tbd ", "x"]})
    out = dd(df, entity_by="e", entity_ignore_values={"TBD"}, action="flag", group_id_column="g", log_stats=False)
    assert ids(out) == [0, 1, 2]


def test_block_by_applies_to_exact_and_group_by_dedupe():
    # Was: block_by silently ignored unless entity_by was used -> cross-tenant data loss.
    df = pd.DataFrame({"tenant": [1, 2, 1], "email": ["a", "a", "a"]})
    assert dd(df, on="email", block_by="tenant", log_stats=False).index.tolist() == [0, 1]
    assert dd(df, group_by="email", block_by="tenant", log_stats=False).index.tolist() == [0, 1]


def test_invalid_scope_is_rejected_not_silently_global():
    with pytest.raises(ValueError, match="entity_scope"):
        dd(pd.DataFrame({"e": ["a"]}), entity_by="e", entity_scope="percolumn", log_stats=False)


def test_empty_on_is_rejected():
    with pytest.raises(ValueError, match="no columns"):
        dd(pd.DataFrame({"e": [1]}), on=[], log_stats=False)


def test_unhashable_exact_key_has_helpful_error():
    df = pd.DataFrame({"tags": [["a"], ["a"]]})
    with pytest.raises(TypeError, match="unhashable"):
        dd(df, log_stats=False)


# --------------------------------------------------------------------------
# Core behaviour
# --------------------------------------------------------------------------

def test_exact_dedupe_matches_pandas_and_keeps_original_index():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.integers(0, 5, 200), "b": rng.choice(["x", "y", None], 200)},
                      index=rng.integers(0, 50, 200))  # duplicate labels
    for keep in ("first", "last", False):
        got = dd(df, keep=keep, log_stats=False)
        exp = df.drop_duplicates(keep=keep)
        pd.testing.assert_frame_equal(got, exp)


def test_transitive_entity_linking_and_scopes():
    df = pd.DataFrame({"email": ["a@x", "A@X ", None, "b@x", "z"],
                       "phone": ["1", None, "1;2", "2", "a@x"]})
    flag = dict(action="flag", group_id_column="g", log_stats=False)
    assert ids(dd(df, entity_by="email phone", **flag)) == [0, 0, 0, 0, 0]
    assert ids(dd(df, entity_by="email phone", entity_scope="per_column", **flag)) == [0, 0, 0, 0, 1]
    assert ids(dd(df, entity_by="email phone", split_entity_values=False, **flag)) == [0, 0, 1, 2, 0]


def test_entity_linking_respects_blocks_and_group_by():
    df = pd.DataFrame({"org": [1, 1, 2, 2], "email": ["a", "a", "a", "b"]})
    flag = dict(action="flag", group_id_column="g", log_stats=False)
    assert ids(dd(df, entity_by="email", block_by="org", **flag)) == [0, 0, 1, 2]
    assert ids(dd(df, entity_by="email", group_by="org", **flag)) == [0, 0, 1, 2]


def test_chained_pick_and_terminal_validation():
    df = pd.DataFrame({"k": [1, 1, 1, 2], "s": [5, 9, 9, 1], "t": [3, 1, 2, 0]})
    assert dd(df, group_by="k", pick=["max:s", "max:t"], log_stats=False).index.tolist() == [2, 3]
    assert dd(df, group_by="k", pick=["max:s", "last"], log_stats=False).index.tolist() == [2, 3]
    with pytest.raises(ValueError, match="last selector"):
        dd(df, group_by="k", pick=["first", "max:s"], log_stats=False)
    with pytest.raises(TypeError, match="numeric"):
        dd(df.astype({"s": str}), group_by="k", pick="max:s", log_stats=False)


def test_max_handles_nullable_and_all_missing_groups():
    df = pd.DataFrame({"k": [1, 1, 2, 2], "v": pd.array([None, 3, None, None], dtype="Int64")})
    assert dd(df, on="k", pick="max:v", log_stats=False).index.tolist() == [1, 2]


def test_random_is_reproducible_and_one_per_group():
    df = pd.DataFrame({"k": np.repeat(np.arange(50), 4)})
    a = dd(df, on="k", pick="random", random_state=1, log_stats=False)
    b = dd(df, on="k", pick="random", random_state=1, log_stats=False)
    assert a.index.equals(b.index) and len(a) == 50 and a.index.is_monotonic_increasing


def test_within_mode_and_flag_output():
    df = pd.DataFrame({"g": [1, 1, 1, 2], "v": ["a", "a", "b", "a"]})
    out, stats = dd(df, group_by="g", on="v", mode="within", action="flag",
                    group_id_column="gid", return_stats=True, log_stats=False)
    assert out["is_duplicate"].tolist() == [False, True, False, False]
    assert out["gid"].tolist() == [0, 0, 0, 1]
    assert stats["flagged_count"] == 1 and stats["mode"] == "within"


def test_group_id_without_grouping_is_duplicate_set_id():
    df = pd.DataFrame({"a": [3, 1, 3]})
    assert ids(dd(df, action="flag", group_id_column="g", log_stats=False)) == [0, 1, 0]


def test_empty_frame():
    df = pd.DataFrame({"email": pd.Series([], dtype=object), "s": pd.Series([], dtype=int)})
    for kw in ({}, {"entity_by": "email"}, {"group_by": "email", "pick": "max:s"}):
        assert dd(df, log_stats=False, **kw).empty


def test_input_is_not_mutated():
    df = pd.DataFrame({"a": [1, 1]})
    snapshot = df.copy()
    dd(df, action="flag", group_id_column="g", log_stats=False)
    dd(df, group_id_column="g", log_stats=False)
    pd.testing.assert_frame_equal(df, snapshot)


# --------------------------------------------------------------------------
# Internals vs reference implementations
# --------------------------------------------------------------------------

def _reference_components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    return pd.factorize(np.array([find(i) for i in range(n)]))[0]


@pytest.mark.parametrize("seed", range(20))
def test_connected_components_matches_union_find(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    e = int(rng.integers(0, 2 * n))
    u, v = rng.integers(0, n, e), rng.integers(0, n, e)
    got = pd.factorize(m._connected_components(n, u, v))[0]
    assert (got == _reference_components(n, zip(u, v))).all()


def test_connected_components_long_shuffled_path():
    n = 200_000
    perm = np.random.default_rng(0).permutation(n)
    labels = m._connected_components(n, perm[:-1], perm[1:])
    assert (labels == 0).all()


def test_vectorised_string_tokens_match_scalar_tokenizer():
    sep = __import__("re").compile(r"[;,|]+")
    values = ["A; b", " c ", "", ";;", "N/A", "x|0|Y", "Straße", None, "  ,a"]
    s = pd.Series(values, dtype=object)
    for coerce in (False, True):
        rows, toks = m._column_tokens(s, sep, DEFAULT_IGNORE, coerce)
        expected = [(i, t) for i, v in enumerate(values) if v is not None
                    for t in m._value_tokens(v, sep, DEFAULT_IGNORE, coerce)]
        assert list(zip(rows.tolist(), toks.tolist())) == expected
