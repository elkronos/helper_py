import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------
# User-friendly column specs
# ----------------------------

ColSpec = Optional[Union[str, Sequence[str]]]


def _split_columns(s: str) -> List[str]:
    """Split on comma, pipe, or whitespace; preserve order; de-dupe."""
    parts: List[str] = []
    for chunk in s.replace("|", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p for p in chunk.split() if p.strip()])

    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _resolve_cols(
    spec: ColSpec, df_cols: Sequence[str], *, name: str, allow_exclude: bool = True
) -> Optional[List[str]]:
    """
    Interprets:
      - None -> None (means: not provided)
      - "*" / "all" / "" -> all columns
      - "A,B" / "A B" / "A|B" -> split
      - "-col1,col2" / "!col1 col2" -> all columns except those (if allow_exclude)
      - list/tuple/set of strings

    Note: column names containing whitespace, commas, or pipes cannot be
    expressed via the string spec (they would be split). Pass such columns
    via the list form, e.g., on=["first name", "a,b"].
    """
    if spec is None:
        return None

    cols_all = list(df_cols)

    if isinstance(spec, str):
        raw = spec.strip()
        if (not raw) or raw.lower() in {"*", "all"}:
            return cols_all

        if allow_exclude and (raw.startswith("-") or raw.startswith("!")):
            excluded = _split_columns(raw[1:])
            invalid = [c for c in excluded if c not in df_cols]
            if invalid:
                raise ValueError(f"Invalid column name(s) in {name} exclude: {invalid}")
            excluded_set = set(excluded)
            return [c for c in cols_all if c not in excluded_set]

        tokens = _split_columns(raw)
        invalid = [c for c in tokens if c not in df_cols]
        if invalid:
            if raw in df_cols:
                raise ValueError(
                    f"Column name '{raw}' contains whitespace; pass a list "
                    f"(e.g., {name}=['{raw}']) instead of a string."
                )
            raise ValueError(f"Invalid column name(s) in {name}: {invalid}")
        return tokens

    if isinstance(spec, (list, tuple, set)):
        tokens = [str(c).strip() for c in spec if str(c).strip()]
        invalid = [c for c in tokens if c not in df_cols]
        if invalid:
            raise ValueError(f"Invalid column name(s) in {name}: {invalid}")
        return tokens

    raise ValueError(f"{name} must be None, a string, or a list/tuple/set of strings.")


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------
# Pick (tie-break) specification
# ----------------------------

Pick = Union[
    str,
    Callable[..., pd.DataFrame],
    List[Union[str, Callable[..., pd.DataFrame]]],
]


@dataclass(frozen=True)
class PickSpec:
    kind: Literal["first", "last", "random", "max", "min", "custom"]
    column: Optional[str] = None


def _parse_pick(pick: Pick) -> Tuple[PickSpec, Optional[Callable[..., pd.DataFrame]]]:
    """
    pick can be:
      - "first" / "last" / "random"
      - "max:col" / "min:col"
      - callable: custom(group, key) -- always called with both arguments;
        accept and ignore `key` if you don't need it.
    """
    if callable(pick):
        return PickSpec(kind="custom"), pick

    raw = str(pick).strip()
    p = raw.lower()

    synonyms = {
        "keep_first": "first",
        "head": "first",
        "first": "first",
        "keep_last": "last",
        "tail": "last",
        "last": "last",
        "rand": "random",
        "random": "random",
    }
    p = synonyms.get(p, p)

    if p.startswith("max:") or p.startswith("min:"):
        kind = "max" if p.startswith("max:") else "min"
        col = raw.split(":", 1)[1].strip()
        if not col:
            raise ValueError("Use pick='max:<col>' or pick='min:<col>'.")
        return PickSpec(kind=kind, column=col), None

    if p in {"first", "last", "random"}:
        return PickSpec(kind=p), None

    raise ValueError("pick must be 'first'/'last'/'random', 'max:<col>'/'min:<col>', or a callable.")


def _call_custom(custom_fn: Callable[..., pd.DataFrame], group: pd.DataFrame) -> pd.DataFrame:
    # Single contract: custom callables always receive (group, key) positionally.
    # Callables that don't need the key should accept it and ignore it. This
    # avoids signature inspection, which is fooled by *args, functools.wraps,
    # and partials.
    return custom_fn(group, getattr(group, "name", None))


def _parse_pandas_version(v: str) -> Tuple[int, int]:
    """Parse 'MAJOR.MINOR(.PATCH)?(suffix)?' into (major, minor); non-numeric suffixes are ignored."""
    parts = v.split(".")

    def _to_int(s: str) -> int:
        m = re.match(r"\d+", s)
        return int(m.group()) if m else 0

    major = _to_int(parts[0]) if len(parts) > 0 else 0
    minor = _to_int(parts[1]) if len(parts) > 1 else 0
    return major, minor


# `include_groups=False` was added in pandas 2.2.0. Detect once at import time
# so we don't conflate a user callable's TypeError with pandas' own signature.
_SUPPORTS_INCLUDE_GROUPS = _parse_pandas_version(pd.__version__) >= (2, 2)


def _groupby_apply(grouped, func):
    """GroupBy.apply wrapper that uses modern behavior when available."""
    if _SUPPORTS_INCLUDE_GROUPS:
        return grouped.apply(func, include_groups=False)
    return grouped.apply(func)


# ----------------------------
# "on" semantics: exact key vs connectivity key
# ----------------------------

@dataclass(frozen=True)
class OnSpec:
    kind: Literal["exact", "any"]  # any = connect if ANY identifier overlaps
    cols: List[str]


_ANY_RE = re.compile(r"^\s*(any|or|link)\s*[: ]\s*(.+)\s*$", flags=re.I)


def _parse_on_spec(on: ColSpec, df_cols: Sequence[str]) -> OnSpec:
    """
    on supports two forms:

      - Exact key (default):
          on="email phone" -> duplicates share the full key (email, phone)

      - Connectivity key:
          on="any: email phone" or "any email phone" -> link if ANY identifier overlaps
          (synonyms: "or:", "link:")

    Notes:
      - any/or/link ignore missing values.
      - list/tuple inputs are treated as exact keys.
    """
    if on is None:
        return OnSpec(kind="exact", cols=list(df_cols))

    if isinstance(on, str):
        raw = on.strip()
        if not raw:
            return OnSpec(kind="exact", cols=list(df_cols))

        m = _ANY_RE.match(raw)
        if m:
            cols = _resolve_cols(m.group(2), df_cols, name="on(any)", allow_exclude=False)
            if not cols:
                raise ValueError("on='any' requires at least one column.")
            return OnSpec(kind="any", cols=cols)

        cols = _resolve_cols(raw, df_cols, name="on", allow_exclude=True)
        return OnSpec(kind="exact", cols=list(df_cols) if cols is None else cols)

    cols = _resolve_cols(on, df_cols, name="on", allow_exclude=True)
    return OnSpec(kind="exact", cols=list(df_cols) if cols is None else cols)


# ----------------------------
# Latent entity grouping via multi-ID connectivity
# ----------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# Sentinel/placeholder values to drop during entity tokenization.
#
# In identity data, values like "0", "-1", "n/a", "unknown", or "" are
# placeholders for "no value" rather than real identifiers. With
# scope="global" they would otherwise act as connecting tokens and silently
# collapse unrelated entities into one giant component. Tokens whose
# canonical form (see `_canonical_token`) appears in this set are dropped.
#
# To replace or extend the default, pass an explicit `ignore=` to
# `_explode_id_value`, e.g.:
#     _explode_id_value(v, split=True, sep_regex=r"[;,|]+",
#                       ignore=DEFAULT_IGNORE | {"placeholder", "tbd"})
DEFAULT_IGNORE: frozenset = frozenset({
    "", "0", "00000000", "n/a", "na", "none", "null", "unknown", "-", "-1",
})


def _type_tag(t: Any) -> str:
    """Return the canonical type prefix used by :func:`_canonical_token`
    (without the trailing ``:``). Kept in lockstep with ``_canonical_token``
    so callers can compute the bare form once and prepend the tag, instead of
    invoking ``_canonical_token`` twice per token in hot loops. ``bool`` must
    be checked before ``int`` because ``bool`` is a subclass of ``int``.
    """
    if isinstance(t, bool):
        return "bool"
    if isinstance(t, (int, np.integer)):
        return "int"
    if isinstance(t, (float, np.floating)):
        return "num"
    return "str"


def _canonical_token(t: Any, *, coerce_token_types: bool = False) -> Optional[str]:
    """
    Coerce a token to a canonical string form for hashing/comparison.

    By default, each token is prefixed with its type (``str:``, ``int:``,
    ``num:``, ``bool:``) so values of different types do not silently collide
    within the same column -- e.g., the int ``123`` and the string ``"123"``
    end up as ``"int:123"`` vs ``"str:123"``. This is the safer default after
    CSV/JSON round-trips where ``True``/``1`` or ``False``/``0`` could
    otherwise be conflated. Note that ``bool`` must be checked before ``int``
    because Python's ``bool`` is a subclass of ``int``.

    For floats, ``format(x, '.17g')`` is used (round-trip safe, unlike
    ``repr``); ``NaN`` returns ``None`` so callers can drop it. Strings are
    ``.strip().casefold()``-ed for case-insensitive comparison across locales.

    Pass ``coerce_token_types=True`` to drop the type prefix when the caller
    explicitly wants cross-type matches (e.g., to treat the int ``123`` and
    the string ``"123"`` as the same identifier after a lossy round-trip).
    """
    if isinstance(t, bool):
        s = "true" if t else "false"
        return s if coerce_token_types else f"bool:{s}"
    if isinstance(t, (int, np.integer)):
        s = str(int(t))
        return s if coerce_token_types else f"int:{s}"
    if isinstance(t, (float, np.floating)):
        f = float(t)
        if math.isnan(f):
            return None
        s = format(f, '.17g')
        return s if coerce_token_types else f"num:{s}"
    s = str(t).strip().casefold()
    return s if coerce_token_types else f"str:{s}"


def _explode_id_value(
    v: Any,
    *,
    split: bool,
    sep_compiled: Optional[re.Pattern],
    ignore: frozenset = DEFAULT_IGNORE,
    strip: bool = True,
    coerce_token_types: bool = False,
) -> List[str]:
    """
    Turns a cell value into zero or more canonical identifier tokens.
      - None/NaN -> []
      - list/tuple/set -> flattened recursively
      - string -> optionally split on separators (e.g., "a;b|c")
      - other scalar -> [value]

    Each surviving token is passed through :func:`_canonical_token`, which by
    default prefixes the token with its type (``str:``, ``int:``, ``num:``,
    ``bool:``) so the int ``123`` and the string ``"123"`` do not silently
    collide after a CSV/JSON round-trip, and ``True``/``1`` /
    ``False``/``0`` cannot accidentally link. Pass ``coerce_token_types=True``
    when cross-type matches are desired (the prefix is dropped and types
    merge).

    Tokens whose un-prefixed canonical form is in `ignore` are dropped so
    that sentinel/placeholder values such as "0", "-1", "n/a", "unknown",
    or "" cannot act as connecting tokens and silently link unrelated
    entities under scope="global". Defaults to `DEFAULT_IGNORE`; pass an
    explicit set to replace or extend it.
    """
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass

    if isinstance(v, (list, tuple, set, frozenset)):
        nested: List[str] = []
        for x in v:
            nested.extend(_explode_id_value(
                x, split=split, sep_compiled=sep_compiled, ignore=ignore, strip=strip,
                coerce_token_types=coerce_token_types,
            ))
        return nested

    if isinstance(v, str):
        s = v.strip() if strip else v
        if not s:
            return []
        if split and sep_compiled is not None and sep_compiled.search(s):
            tokens = [p.strip() for p in sep_compiled.split(s) if p and p.strip()]
        else:
            tokens = [s]
    else:
        tokens = [v]

    # Canonicalize at the bottom so callers always receive hashable,
    # type-stable strings. Compute the *un-prefixed* canonical form once and
    # check it against `ignore` before re-attaching the type tag, so we don't
    # call `_canonical_token` twice per token in this hot loop. Ignore-set
    # membership is checked against the bare form so DEFAULT_IGNORE entries
    # like "0" or "n/a" still match regardless of whether the original value
    # arrived as an int, float, or string after a CSV/JSON round-trip.
    out: List[str] = []
    for t in tokens:
        bare = _canonical_token(t, coerce_token_types=True)
        if bare is None or bare in ignore:
            continue
        out.append(bare if coerce_token_types else f"{_type_tag(t)}:{bare}")
    return out


def _group_ids_from_entity_by(
    df: pd.DataFrame,
    entity_cols: List[str],
    *,
    scope: Literal["global", "per_column"] = "global",
    split_values: bool = True,
    sep_regex: str = r"[;,|]+",
    ignore: frozenset = DEFAULT_IGNORE,
    coerce_token_types: bool = False,
) -> pd.Series:
    """
    Build entity groups by connectivity across identifier tokens.

    Two rows are connected if they share ANY identifier token within entity_cols.
      - scope="global": token equality across columns connects (email and alt_email connect)
      - scope="per_column": only same-column matches connect (safer for ID namespaces)
    """
    n = len(df)
    uf = _UnionFind(n)
    first_seen: Dict[Any, int] = {}
    sep_compiled = re.compile(sep_regex) if split_values else None

    for col in entity_cols:
        values = df[col].to_numpy(dtype=object, copy=False)
        for i in range(n):
            tokens = _explode_id_value(
                values[i], split=split_values, sep_compiled=sep_compiled, ignore=ignore,
                coerce_token_types=coerce_token_types,
            )
            for t in tokens:
                key = (col, t) if scope == "per_column" else t
                j = first_seen.get(key)
                if j is None:
                    first_seen[key] = i
                else:
                    uf.union(i, j)

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    codes, _ = pd.factorize(roots, sort=False)
    return pd.Series(codes, index=df.index, name="_group_id")


def _group_ids_from_group_by(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    keys = pd.MultiIndex.from_frame(df[group_cols], names=group_cols)
    codes, _ = pd.factorize(keys, sort=False)
    return pd.Series(codes, index=df.index, name="_group_id")


def _auto_entity_cols(df: pd.DataFrame) -> List[str]:
    """
    Conservative heuristic for identifier-like columns:
      - name suggests identity (email/phone/id/uuid/guid/identifier)
      - prefer object/string columns; allow numeric if name strongly indicates an ID
    """
    name_pat = re.compile(
        r"(email|e[-_ ]?mail|phone|mobile|msisdn|uuid|guid|identifier|external[_ ]?id|"
        r"customer[_ ]?id|user[_ ]?id|account[_ ]?id|(^|[_ ])id($|[_ ]))",
        re.I,
    )

    cols: List[str] = []
    for c in df.columns:
        cn = str(c)
        if not name_pat.search(cn):
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            cols.append(cn)
        else:
            if re.search(r"(uuid|guid|(^|[_ ])id($|[_ ]))", cn, flags=re.I):
                cols.append(cn)

    return _unique_preserve_order(cols)


# ----------------------------
# Selection helpers
# ----------------------------

def _select_extreme_row(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    mode: Literal["max", "min"],
) -> pd.DataFrame:
    """
    Select one row per group based on the extreme value of value_col.
    If all values are missing within a group, selection falls back to the first row in that group.
    """
    gb = df.groupby(group_cols, sort=False)[value_col]
    extreme = gb.transform("max" if mode == "max" else "min")

    v = df[value_col]
    mask = v.eq(extreme) | (v.isna() & extreme.isna())
    candidates = df.loc[mask]
    return candidates.drop_duplicates(subset=group_cols, keep="first")


def _reorder_like_input(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """Stable output order aligned to the input DataFrame."""
    pos = pd.Index(df_in.index).get_indexer(df_out.index)
    order = np.argsort(pos, kind="mergesort")
    return df_out.loc[df_out.index[order]]


def _apply_pick(
    df: pd.DataFrame,
    pick: Union[str, Callable[..., pd.DataFrame]],
    key_cols: List[str],
    *,
    allow_ties: bool = False,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reduce df to a per-group survivor set defined by key_cols, according to `pick`.

    When allow_ties=True (used by intermediate selectors in a chained pick),
    rows tied at the chosen max/min are all kept so the next selector can
    break the tie. "first"/"last"/"random" inherently pick a single row, so
    allow_ties has no effect on them -- they're terminal selectors.
    """
    pick_spec, pick_fn = _parse_pick(pick)

    if pick_spec.kind == "first":
        return df.drop_duplicates(subset=key_cols, keep="first")

    if pick_spec.kind == "last":
        return df.drop_duplicates(subset=key_cols, keep="last")

    if pick_spec.kind == "random":
        gb = df.groupby(key_cols, sort=False, group_keys=False)
        try:
            out = gb.sample(n=1, random_state=random_state)
        except Exception:
            out = _groupby_apply(
                gb, lambda g: df.loc[g.index].sample(n=1, random_state=random_state)
            )
        return _reorder_like_input(df, out)

    if pick_spec.kind in {"max", "min"}:
        assert pick_spec.column is not None
        if allow_ties:
            gb = df.groupby(key_cols, sort=False)[pick_spec.column]
            extreme = gb.transform("max" if pick_spec.kind == "max" else "min")
            v = df[pick_spec.column]
            mask = v.eq(extreme) | (v.isna() & extreme.isna())
            return df.loc[mask]
        return _select_extreme_row(df, key_cols, pick_spec.column, pick_spec.kind)  # type: ignore[arg-type]

    # custom callable
    assert pick_fn is not None
    gb = df.groupby(key_cols, sort=False, group_keys=False)
    out = _groupby_apply(gb, lambda g: _call_custom(pick_fn, df.loc[g.index]))
    return _reorder_like_input(df, out)


def _apply_pick_chain(
    df: pd.DataFrame,
    pick: Pick,
    key_cols: List[str],
    *,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Dispatch single-pick or list-of-picks to _apply_pick, applying chained
    picks left-to-right on the shrinking survivor set."""
    if isinstance(pick, list):
        if not pick:
            raise ValueError("pick=[] is not allowed; provide at least one selector.")
        # Apply each pick in sequence on the survivor set; the last one must
        # produce a unique survivor per group (or use "first" implicitly).
        # Reject terminal selectors ("first"/"last"/"random") in non-final
        # positions: they each pick exactly one row, so any later selector
        # would be a silent no-op -- almost certainly a user error.
        keep_df = df
        for sub_pick in pick[:-1]:
            sub_spec, _ = _parse_pick(sub_pick)
            if sub_spec.kind in {"first", "last", "random"}:
                raise ValueError(
                    f"pick='{sub_spec.kind}' is a terminal selector and must be "
                    f"the last element of a chained pick (it picks one row per "
                    f"group, so later selectors would be no-ops); "
                    f"move it to the end or remove it."
                )
            keep_df = _apply_pick(keep_df, sub_pick, key_cols, allow_ties=True, random_state=random_state)
        keep_df = _apply_pick(keep_df, pick[-1], key_cols, allow_ties=False, random_state=random_state)
        return keep_df
    return _apply_pick(df, pick, key_cols, allow_ties=False, random_state=random_state)


# ----------------------------
# Main API
# ----------------------------

# Sentinel used to detect whether the caller explicitly passed `pick`.
# Needed so the `keep=` alias can't silently override an explicit `pick=`.
_UNSET: Any = object()


def deduplicate_dataframe(
    df: pd.DataFrame,
    on: ColSpec = None,
    *,
    action: str = "remove",
    pick: Pick = _UNSET,
    group_by: ColSpec = None,
    entity_by: Union[ColSpec, Literal["auto"], bool] = None,
    block_by: ColSpec = None,
    mode: str = "auto",
    random_state: Optional[int] = None,
    flag_column: str = "is_duplicate",
    group_id_column: Optional[str] = None,
    return_stats: bool = False,
    log_stats: bool = True,
    # connectivity behavior
    entity_scope: Literal["global", "per_column"] = "global",
    split_entity_values: bool = True,
    entity_separators_regex: str = r"[;,|]+",
    entity_ignore_values: Optional[Iterable] = None,
    coerce_token_types: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Deduplicate with optional entity consolidation for datasets with multiple identifiers.

    Inputs are forgiving:
      - on / group_by / entity_by / block_by accept "A,B" or "A B" or "A|B"
      - on supports exclusion: "-timestamp" means "all columns except timestamp"
      - on supports connectivity: "any: email phone" (or "any email phone", "or:", "link:")

    Behavior:
      - No grouping: remove exact duplicates on `on` using `pick`.
      - group_by: explicit grouping keys.
      - entity_by: infer latent entities by linking rows that share any identifier token.
      - block_by: restrict linking within blocks (e.g., tenant_id / org_id).
      - If both group_by and entity_by are provided, group_by is treated as an additional block constraint.

    mode:
      - auto: collapse if grouped (group_by/entity_by or on uses connectivity), otherwise within
      - collapse: one row per group/entity (consolidation)
      - within: dedupe (by exact `on` columns) within each group/entity (or globally if not grouped)

    pick:
      - "first" / "last" / "random"
      - "max:col" / "min:col"
      - callable(group, key)->DataFrame -- always called with both arguments
        positionally; callables that don't need the key should accept it and
        ignore it (e.g., `def pick_fn(group, _key): ...`).
        NOTE: This is a breaking change from earlier versions, which also
        accepted a one-arg callable(group)->DataFrame. Single-arg callables
        will now raise TypeError; wrap them as `lambda g, _k: fn(g)` to adapt.
      - list of any of the above for chained tie-breakers, applied
        left-to-right on the shrinking survivor set
        (e.g., ["max:score", "max:timestamp"] = max score, then latest
        timestamp; the last selector must reduce each group to one row,
        so use "first" implicitly if needed).

    group_id_column:
      Optional name for an output column carrying the group/entity id assigned
      during grouping. Note that this column's coverage differs by action: in
      action="remove" mode only the surviving (kept) rows carry the gid because
      removed rows are no longer present, while in action="flag" mode every row
      (kept and flagged alike) carries its gid so callers can inspect group
      membership for duplicates as well.

    entity_ignore_values:
      Optional iterable of canonical (un-prefixed, casefolded) token strings to
      treat as placeholders during entity linking. When None (default), uses
      DEFAULT_IGNORE ({"", "0", "n/a", "none", "unknown", "-1", ...}). Pass an
      explicit set to replace it, e.g. `DEFAULT_IGNORE | {"placeholder", "tbd"}`
      to extend, or `frozenset()` to disable placeholder filtering entirely.

    coerce_token_types:
      If False (default), entity tokens are prefixed with their type tag
      (``str:``, ``int:``, ``num:``, ``bool:``) so the int ``123`` and the
      string ``"123"`` cannot silently link after a CSV/JSON round-trip. Set
      True to drop the prefix when cross-type matches are intentional.
    """
    # Optional aliases for short, common patterns.
    if "subset" in kwargs and on is None:
        on = kwargs.pop("subset")
    if "cols" in kwargs and on is None:
        on = kwargs.pop("cols")

    if "by" in kwargs and group_by is None:
        group_by = kwargs.pop("by")
    if "group" in kwargs and group_by is None:
        group_by = kwargs.pop("group")

    if "ids" in kwargs and entity_by is None:
        entity_by = kwargs.pop("ids")
    if "entity" in kwargs and entity_by is None:
        entity_by = kwargs.pop("entity")

    if "block" in kwargs and block_by is None:
        block_by = kwargs.pop("block")

    if "keep" in kwargs:
        if pick is not _UNSET:
            raise TypeError("Pass either pick= or keep=, not both.")
        pick = kwargs.pop("keep")
    if pick is _UNSET:
        pick = "first"

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unknown argument(s): {unknown}")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    start = time.perf_counter()

    # Normalize the index for all internal work: a unique RangeIndex makes
    # get_indexer / isin / loc / groupby safe even when the caller's index has
    # duplicate labels. The original index is restored on the way out.
    _original_index = df.index
    df = df.reset_index(drop=True)

    # Normalize action
    a = str(action).strip().lower()
    if a in {"remove", "rm", "drop", "delete"}:
        action_n: Literal["remove", "flag"] = "remove"
    elif a in {"flag", "mark", "tag"}:
        action_n = "flag"
    else:
        raise ValueError("action must be 'remove'/'flag' (or a common synonym).")

    # Validate output column names won't collide with existing columns.
    if action_n == "flag" and flag_column in df.columns:
        raise ValueError(f"flag_column '{flag_column}' already exists in df.")
    if group_id_column is not None and group_id_column in df.columns:
        raise ValueError(f"group_id_column '{group_id_column}' already exists in df.")
    if action_n == "flag" and group_id_column == flag_column:
        raise ValueError("flag_column and group_id_column must differ.")

    # Parse on spec (exact vs connectivity)
    on_spec = _parse_on_spec(on, df.columns)

    # Normalize mode
    m = str(mode).strip().lower()
    if m in {"auto", "a"}:
        mode_n: Literal["auto", "collapse", "within"] = "auto"
    elif m in {"collapse", "c", "one"}:
        mode_n = "collapse"
    elif m in {"within", "w"}:
        mode_n = "within"
    else:
        raise ValueError("mode must be 'auto'/'collapse'/'within' (or a common synonym).")

    # Resolve group_by, entity_by, block_by
    group_cols_input = _resolve_cols(group_by, df.columns, name="group_by", allow_exclude=False)

    if entity_by is True or (isinstance(entity_by, str) and entity_by.strip().lower() == "auto"):
        entity_cols = _auto_entity_cols(df)
        if not entity_cols:
            raise ValueError("entity_by='auto' did not find any identifier-like columns.")
    else:
        entity_cols = _resolve_cols(
            entity_by if entity_by is not False else None, df.columns, name="entity_by", allow_exclude=False
        )

    block_cols = _resolve_cols(block_by, df.columns, name="block_by", allow_exclude=False)

    # If on uses connectivity and neither group_by nor entity_by are provided, infer entities from on cols.
    if on_spec.kind == "any" and entity_cols is None:
        entity_cols = on_spec.cols

    # If both group_by and entity_by are present, treat group_by as additional block constraints.
    # This is a common identity-data pattern (link identifiers within tenant/org/customer partitions).
    if entity_cols is not None and group_cols_input is not None:
        block_cols = _unique_preserve_order((block_cols or []) + list(group_cols_input))
        group_cols_for_grouping: Optional[List[str]] = None
    else:
        group_cols_for_grouping = group_cols_input

    # Validate pick (single or list); each max:/min: element must reference a
    # column present in df with a numeric/datetime/boolean dtype.
    if isinstance(pick, list):
        if not pick:
            raise ValueError("pick=[] is not allowed; provide at least one selector.")
        _pick_items: List[Union[str, Callable[..., pd.DataFrame]]] = list(pick)
    else:
        _pick_items = [pick]
    for _item in _pick_items:
        _spec, _ = _parse_pick(_item)
        if _spec.kind in {"max", "min"}:
            assert _spec.column is not None
            if _spec.column not in df.columns:
                raise ValueError(f"pick column '{_spec.column}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[_spec.column]) \
               and not pd.api.types.is_datetime64_any_dtype(df[_spec.column]) \
               and not pd.api.types.is_bool_dtype(df[_spec.column]):
                raise TypeError(
                    f"pick='{_spec.kind}:{_spec.column}' requires a numeric, "
                    f"datetime, or boolean column; got {df[_spec.column].dtype}."
                )

    is_grouped = (group_cols_for_grouping is not None) or (entity_cols is not None)
    if mode_n == "auto":
        mode_n = "collapse" if is_grouped else "within"

    original_count = len(df)
    work = df

    # ---- Build group ids (optional) ----
    group_id: Optional[pd.Series] = None
    grouping_kind: Literal["none", "group_by", "entity_by"] = "none"

    if group_cols_for_grouping is not None:
        grouping_kind = "group_by"
        group_id = _group_ids_from_group_by(work, group_cols_for_grouping)

    elif entity_cols is not None:
        grouping_kind = "entity_by"
        # Resolve the user-supplied ignore set to a frozenset once; default to
        # DEFAULT_IGNORE when the caller didn't provide one.
        entity_ignore = (
            DEFAULT_IGNORE if entity_ignore_values is None
            else frozenset(entity_ignore_values)
        )
        if block_cols:
            parts: List[pd.Series] = []
            offset = 0
            for _, block in work.groupby(block_cols, sort=False, dropna=False):
                local_gid = _group_ids_from_entity_by(
                    block,
                    entity_cols,
                    scope=entity_scope,
                    split_values=split_entity_values,
                    sep_regex=entity_separators_regex,
                    ignore=entity_ignore,
                    coerce_token_types=coerce_token_types,
                )
                local_gid = local_gid + offset
                offset = int(local_gid.max()) + 1 if len(local_gid) else offset
                parts.append(local_gid)
            group_id = pd.concat(parts).reindex(work.index)
        else:
            group_id = _group_ids_from_entity_by(
                work,
                entity_cols,
                scope=entity_scope,
                split_values=split_entity_values,
                sep_regex=entity_separators_regex,
                ignore=entity_ignore,
                coerce_token_types=coerce_token_types,
            )

    # Effective exact columns for "within" mode.
    # If on was connectivity-based, within mode uses the same column list as an exact key.
    on_cols_exact = on_spec.cols

    # ---- Compute rows to keep ----
    if group_id is None:
        # Global (no grouping): exact-key dedupe on on_cols_exact
        keep_df = _apply_pick_chain(work, pick, on_cols_exact, random_state=random_state)

    else:
        # Grouped behavior. The gid is an internal implementation detail; pick a
        # name that's guaranteed not to clobber a user column (e.g., a df that
        # already has a column literally named "_group_id"). Note that
        # `group_id.name` is always "_group_id" today -- both _group_ids_from_*
        # builders hardcode it -- so we don't bother consulting it here.
        gid_name = "_group_id"
        while gid_name in work.columns:
            gid_name = "_" + gid_name
        work2 = work.copy()
        work2[gid_name] = group_id

        if mode_n == "collapse":
            # One row per group/entity (consolidation)
            keep_df = _apply_pick_chain(work2, pick, [gid_name], random_state=random_state)

        else:
            # within: exact-key dedupe within each group/entity using on_cols_exact
            key_cols = _unique_preserve_order([gid_name] + list(on_cols_exact))
            keep_df = _apply_pick_chain(work2, pick, key_cols, random_state=random_state)

        if group_id_column is None:
            keep_df = keep_df.drop(columns=[gid_name])
        else:
            keep_df = keep_df.rename(columns={gid_name: group_id_column})

    # ---- Produce output ----
    if action_n == "remove":
        out = keep_df
    else:
        out = work.copy()
        out[flag_column] = ~out.index.isin(keep_df.index)
        if group_id is not None and group_id_column is not None:
            out[group_id_column] = group_id

    # Restore the caller's original index. In "flag" mode `out` has one row per
    # input row (same positional index), so the original index maps directly.
    # In "remove" mode `out` is a subset; index it positionally into the
    # original index to recover the labels for the surviving rows.
    if action_n == "flag":
        out.index = _original_index
    else:
        out.index = _original_index[out.index]

    # ---- Stats ----
    kept_count = int(len(keep_df))
    removed_count = int(original_count - kept_count)
    flagged_count = int(out[flag_column].sum()) if action_n == "flag" else 0

    stats = {
        "original_count": int(original_count),
        "kept_count": kept_count,
        "removed_count": removed_count if action_n == "remove" else 0,
        "flagged_count": flagged_count,
        "grouping": grouping_kind,
        "mode": mode_n,
        "pick": (
            [p if isinstance(p, str) else getattr(p, "__name__", "custom_callable") for p in pick]
            if isinstance(pick, list)
            else (pick if isinstance(pick, str) else getattr(pick, "__name__", "custom_callable"))
        ),
        "on_kind": on_spec.kind,
        "on_cols": on_spec.cols,
        "entity_by": entity_cols if grouping_kind == "entity_by" else None,
        "group_by": group_cols_for_grouping if grouping_kind == "group_by" else None,
        "block_by": block_cols if block_cols else None,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if log_stats:
        logger.info("Deduplication stats: %s", stats)

    return (out, stats) if return_stats else out


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    data = {
        "customer_id": [100, 100, 101, 102, 102, 103],
        "email": ["a@x.com", "a@x.com", "b@x.com", "c@x.com", None, "c@x.com"],
        "phone": ["555-1", None, "555-2", "555-3", "555-3", None],
        "score": [10, 12, 7, 20, 19, 18],
        "payload": ["v1", "v2", "v3", "v4", "v5", "v6"],
    }
    df = pd.DataFrame(data)

    # Exact-key dedupe on all columns (exact duplicates only)
    print(deduplicate_dataframe(df))

    # Exact-key dedupe on the pair (email, phone)
    print(deduplicate_dataframe(df, on="email phone"))

    # Connectivity dedupe (email OR phone) as a shorthand
    print(deduplicate_dataframe(df, on="any email phone", pick="max:score", group_id_column="entity_id"))

    # Explicit group collapse
    print(deduplicate_dataframe(df, group_by="customer_id", pick="max:score"))

    # Entity linking within an explicit block (customer_id used as a block constraint)
    print(deduplicate_dataframe(df, group_by="customer_id", entity_by="email phone", pick="max:score", group_id_column="entity_id"))

    # Flag rows that would be removed in connectivity collapse
    print(deduplicate_dataframe(df, on="any: email phone", pick="max:score", action="flag", group_id_column="entity_id"))

    # Auto-detect identifier columns (conservative)
    # print(deduplicate_dataframe(df, entity_by="auto", pick="max:score", group_id_column="entity_id"))