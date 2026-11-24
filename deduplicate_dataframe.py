"""
Flexible DataFrame de-duplication with optional entity resolution.

``deduplicate_dataframe`` covers three jobs behind one forgiving API:

* exact de-duplication on a column key (``drop_duplicates`` with richer
  survivor selection),
* consolidation of explicit groups (``group_by``), and
* entity resolution: rows sharing *any* identifier token (email, phone, ...)
  are linked transitively into one entity (``entity_by`` / ``on="any: ..."``).

Every strategy reduces to the same core: compute one integer group code per
row, then run the ``pick`` selectors over those codes. All work is
positional, so the caller's index (duplicate labels included) passes through
untouched and nothing is copied until the output is built.

Missing values in key columns compare equal (as in ``drop_duplicates``) for
every selector; missing values never link entities.

Requires pandas >= 1.5.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ColSpec = Optional[Union[str, Sequence[Hashable]]]
PickFn = Callable[[pd.DataFrame, Any], Any]
PickItem = Union[str, bool, Tuple[str, Hashable], PickFn]
Pick = Union[PickItem, List[PickItem]]

_UNSET: Any = object()  # detects whether `pick` was passed explicitly


# ---------------------------------------------------------------------------
# Column specs
# ---------------------------------------------------------------------------

def _split_columns(s: str) -> List[str]:
    """Split on commas, pipes or whitespace; drop empties; de-dupe in order."""
    return list(dict.fromkeys(t for t in re.split(r"[\s,|]+", s) if t))


def _resolve_cols(
    spec: ColSpec, df: pd.DataFrame, *, name: str, allow_exclude: bool = False
) -> Optional[List[Hashable]]:
    """
    Resolve a column spec against ``df``:

      None                      -> None (not provided)
      "" / "*" / "all"          -> every column
      "a,b" / "a b" / "a|b"     -> those columns
      "-a,b" / "!a b"           -> every column except those (if allow_exclude)
      list/tuple/set/Index      -> those exact labels (any hashable, no parsing)

    A string that exactly names a column wins over splitting, so
    ``on="first name"`` works when that column exists.
    """
    if spec is None:
        return None
    columns = df.columns

    exclude = False
    if isinstance(spec, str):
        raw = spec.strip()
        if raw.lower() in {"", "*", "all"}:
            return list(columns)
        if raw in columns:
            names: List[Hashable] = [raw]
        else:
            exclude = allow_exclude and raw[0] in "-!"
            names = list(_split_columns(raw[1:] if exclude else raw))
    elif isinstance(spec, (list, tuple, set, frozenset, pd.Index)):
        names = list(dict.fromkeys(spec))
    else:
        raise TypeError(f"{name} must be None, a string, or a sequence of column labels.")

    missing = [c for c in names if c not in columns]
    if missing:
        raise ValueError(f"Unknown column(s) in {name}: {missing}")
    if not columns.is_unique:
        ambiguous = [c for c in names if columns.get_indexer_for([c]).size > 1]
        if ambiguous:
            raise ValueError(f"Column label(s) in {name} are duplicated in df: {ambiguous}")

    if exclude:
        dropped = set(names)
        return [c for c in columns if c not in dropped]
    return names


_ANY_RE = re.compile(r"^\s*(any|or|link)\s*(:|\s)\s*(.+?)\s*$", re.I)


def _parse_on(on: ColSpec, df: pd.DataFrame) -> Tuple[bool, List[Hashable]]:
    """
    Return ``(connect_any, columns)``.

    ``"any: email phone"`` (synonyms ``or``/``link``; the colon is optional)
    links rows sharing ANY identifier. Without a colon, the prefix is only
    read as a keyword when it is not itself a column, so a real column named
    ``link`` keeps its exact-key meaning.
    """
    if isinstance(on, str) and on.strip() not in df.columns:
        m = _ANY_RE.match(on)
        if m and (m.group(2) == ":" or m.group(1) not in df.columns):
            return True, _resolve_cols(m.group(3), df, name="on(any)")  # type: ignore[return-value]

    cols = _resolve_cols(on, df, name="on", allow_exclude=True)
    if cols is None:
        cols = list(df.columns)
    if not cols:
        raise ValueError("on resolves to no columns.")
    return False, cols


def _key_values(df: pd.DataFrame, cols: List[Hashable], pos: int) -> List[Any]:
    return [df[c].iat[pos] for c in cols]


# ---------------------------------------------------------------------------
# Integer group codes
# ---------------------------------------------------------------------------

def _codes(s: pd.Series) -> np.ndarray:
    """Dense codes for one column; missing values form their own code."""
    try:
        return pd.factorize(s, use_na_sentinel=False)[0].astype(np.int64, copy=False)
    except TypeError as exc:
        raise TypeError(
            f"Column {s.name!r} holds unhashable values (e.g. lists) and cannot be "
            f"part of an exact key. Exclude it from `on` (e.g. on='-{s.name}') or "
            f"use it for entity linking instead."
        ) from exc


def _combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense codes for the pair (a, b) of non-negative code arrays, in first-seen order."""
    if not len(a):
        return a
    return pd.factorize(a * (int(b.max()) + 1) + b)[0].astype(np.int64, copy=False)


def _key_codes(
    df: pd.DataFrame, cols: Sequence[Hashable], base: Optional[np.ndarray] = None
) -> np.ndarray:
    """One code per distinct combination of ``base`` and the values in ``cols``."""
    codes = np.zeros(len(df), np.int64) if base is None else base
    for c in cols:
        codes = _combine(codes, _codes(df[c]))
    return codes


# ---------------------------------------------------------------------------
# Entity linking: identifier tokens + connected components
# ---------------------------------------------------------------------------

# Placeholder values that mean "no value" rather than a real identifier. Under
# global linking they would otherwise connect unrelated rows into one giant
# entity. Compared against the bare (casefolded, untyped) token.
# Extend via `entity_ignore_values=DEFAULT_IGNORE | {"tbd"}`.
DEFAULT_IGNORE: frozenset = frozenset({
    "", "-", "-1", "0", "00000000", "#n/a", "<na>", "n/a", "na", "nan", "nat",
    "none", "null", "unknown",
})

_CONTAINERS = (list, tuple, set, frozenset, np.ndarray)


def _canonical(v: Any) -> Tuple[str, str]:
    """(type tag, bare form) for a non-missing, non-container scalar.

    Numbers share one ``num`` tag and compare by value, so ``123``, ``123.0``
    and ``np.int32(123)`` agree even after an int column is upcast to float
    by a missing value. ``bool`` is checked first: it subclasses ``int``.
    """
    if isinstance(v, (bool, np.bool_)):
        return "bool", "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return "num", str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "num", str(int(f)) if f.is_integer() else repr(f)
    return "str", str(v).strip().casefold()


def _isna(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _value_tokens(
    v: Any, sep: Optional[re.Pattern], ignore: frozenset, coerce: bool
) -> List[str]:
    """Canonical identifier tokens for one cell (flattening containers)."""
    if isinstance(v, str):
        parts = sep.split(v) if sep is not None else [v]
        bares = [p.strip().casefold() for p in parts]
        return [b if coerce else "str:" + b for b in bares if b and b not in ignore]
    if isinstance(v, _CONTAINERS):
        return [t for x in v for t in _value_tokens(x, sep, ignore, coerce)]
    if v is None or _isna(v):
        return []
    tag, bare = _canonical(v)
    if not bare or bare in ignore:
        return []
    return [bare if coerce else f"{tag}:{bare}"]


def _expand(codes: np.ndarray, lists: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Turn per-row codes into per-distinct-value token lists -> flat (row, token) pairs."""
    lens = np.fromiter(map(len, lists), np.int64, len(lists))
    flat = np.empty(int(lens.sum()), dtype=object)
    flat[:] = [t for lst in lists for t in lst]
    row_lens = lens[codes]
    rows = np.repeat(np.arange(len(codes)), row_lens)
    starts = np.cumsum(lens) - lens
    within = np.arange(len(rows)) - np.repeat(np.cumsum(row_lens) - row_lens, row_lens)
    return rows, flat[np.repeat(starts[codes], row_lens) + within]


def _column_tokens(
    s: pd.Series, sep: Optional[re.Pattern], ignore: frozenset, coerce: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """(row positions, tokens) for one identifier column."""
    def tokenize(v: Any) -> List[str]:
        return _value_tokens(v, sep, ignore, coerce)

    if s.dtype == object and pd.api.types.infer_dtype(s, skipna=True) != "string":
        # Mixed objects: memoise per (type, value), since factorize would
        # merge 1, 1.0 and True into one value.
        codes = np.empty(len(s), np.int64)
        memo: Dict[Any, int] = {}
        lists: List[List[str]] = []
        for i, v in enumerate(s.to_numpy()):
            try:
                key: Any = (v.__class__, v)
                k = memo.get(key)
            except TypeError:  # unhashable cell (list, ndarray, ...)
                key, k = None, None
            if k is None:
                k = len(lists)
                lists.append(tokenize(v))
                if key is not None:
                    memo[key] = k
            codes[i] = k
    else:
        # Homogeneous values (all strings, or a typed column): tokenise each
        # distinct value once.
        codes, uniques = pd.factorize(s)
        lists = [tokenize(u) for u in uniques] + [[]]
        codes = np.where(codes < 0, len(lists) - 1, codes)

    return _expand(codes, lists)


def _connected_components(n_nodes: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Label each node with the smallest node id in its component.

    Vectorised union-find: hook every unsatisfied edge's larger root onto the
    smaller one, then pointer-jump until every node points at its root. Roots
    only ever point to smaller ids, so no cycles form and each round strictly
    reduces the number of roots.
    """
    parent = np.arange(n_nodes)
    while len(u):
        pu, pv = parent[u], parent[v]
        live = pu != pv
        if not live.any():
            break
        u, v, pu, pv = u[live], v[live], pu[live], pv[live]
        np.minimum.at(parent, np.maximum(pu, pv), np.minimum(pu, pv))
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
    return parent


def _entity_codes(
    df: pd.DataFrame,
    cols: Sequence[Hashable],
    *,
    blocks: Optional[np.ndarray],
    per_column: bool,
    sep: Optional[re.Pattern],
    ignore: frozenset,
    coerce: bool,
) -> np.ndarray:
    """Entity id per row: rows sharing any token (within a block) are linked transitively."""
    n = len(df)
    pairs = [_column_tokens(df[c], sep, ignore, coerce) for c in cols]
    rows = np.concatenate([r for r, _ in pairs])
    if not len(rows):
        return np.arange(n, dtype=np.int64)

    token = pd.factorize(np.concatenate([t for _, t in pairs]))[0].astype(np.int64)
    if per_column:
        token = _combine(token, np.repeat(np.arange(len(cols)), [len(r) for r, _ in pairs]))
    if blocks is not None:
        token = _combine(token, blocks[rows])

    # Bipartite graph: row nodes [0, n), token nodes [n, n + n_tokens).
    labels = _connected_components(n + int(token.max()) + 1, rows, n + token)
    return pd.factorize(labels[:n])[0].astype(np.int64)


_AUTO_ENTITY_RE = re.compile(
    r"""^(?:
        (?:(?:alt|alternate|other|primary|secondary|work|home|personal|business)_)?
        (?:e_?mail(?:_?address)?|phone(?:_?(?:number|no))?|mobile(?:_?number)?|cell(?:_?phone)?|msisdn)
      | (?:customer|user|account|member|person|client|contact|patient|external)_?(?:id|uuid|guid|key|number|no)
      | id | uuid | guid
    )_?\d*$""",
    re.I | re.X,
)


def _auto_entity_cols(df: pd.DataFrame, exclude: Iterable[Hashable] = ()) -> List[Hashable]:
    """
    Conservative pick of identity columns: emails/phones and ids of a *person*
    (customer_id, user_id, ...). Generic foreign keys such as ``product_id`` or
    ``store_id`` are deliberately excluded: linking on them would merge every
    customer who bought the same product into one entity.
    """
    skip = set(exclude)
    return [
        c for c in df.columns
        if c not in skip and _AUTO_ENTITY_RE.match(re.sub(r"[\s\-.]+", "_", str(c).strip()))
    ]


# ---------------------------------------------------------------------------
# Survivor selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Selector:
    kind: str  # first | last | none | random | max | min | custom
    column: Hashable = None
    fn: Optional[PickFn] = None

    @property
    def terminal(self) -> bool:
        """Picks at most one row per group, so nothing after it could matter."""
        return self.kind in {"first", "last", "none", "random"}

    @property
    def label(self) -> Any:
        if self.kind == "custom":
            return getattr(self.fn, "__name__", "custom_callable")
        return f"{self.kind}:{self.column}" if self.column is not None else self.kind


_PICK_SYNONYMS = {
    "first": "first", "keep_first": "first", "head": "first",
    "last": "last", "keep_last": "last", "tail": "last",
    "random": "random", "rand": "random",
    "none": "none", "keep_none": "none", "drop_all": "none",
}


def _parse_pick(pick: PickItem) -> _Selector:
    if pick is False:  # pandas' keep=False
        return _Selector("none")
    if callable(pick):
        return _Selector("custom", fn=pick)
    if isinstance(pick, tuple) and len(pick) == 2 and str(pick[0]).lower() in {"max", "min"}:
        return _Selector(str(pick[0]).lower(), column=pick[1])  # any hashable label
    if isinstance(pick, str):
        head, colon, col = pick.strip().partition(":")
        kind = head.strip().lower()
        if colon and kind in {"max", "min"}:
            if not col.strip():
                raise ValueError("Use pick='max:<col>' or pick='min:<col>'.")
            return _Selector(kind, column=col.strip())
        if not colon and kind in _PICK_SYNONYMS:
            return _Selector(_PICK_SYNONYMS[kind])
    raise ValueError(
        "pick must be 'first'/'last'/'random'/'none' (or False), 'max:<col>'/'min:<col>', "
        f"('max'|'min', col), a callable, or a list of these; got {pick!r}."
    )


def _orderable(s: pd.Series) -> bool:
    dt = s.dtype
    return (
        pd.api.types.is_numeric_dtype(dt)
        or pd.api.types.is_datetime64_any_dtype(dt)
        or pd.api.types.is_timedelta64_dtype(dt)
        or (isinstance(dt, pd.CategoricalDtype) and bool(dt.ordered))
    )


def _parse_picks(pick: Pick, df: pd.DataFrame) -> List[_Selector]:
    selectors = [_parse_pick(p) for p in (pick if isinstance(pick, list) else [pick])]
    if not selectors:
        raise ValueError("pick=[] is not allowed; provide at least one selector.")
    for s in selectors[:-1]:
        if s.terminal:
            raise ValueError(
                f"pick='{s.kind}' keeps at most one row per group, so it must be the "
                f"last selector in a chain (later selectors would be no-ops)."
            )
    for s in selectors:
        if s.column is None:
            continue
        if s.column not in df.columns:
            raise ValueError(f"pick column {s.column!r} not found in DataFrame.")
        if not _orderable(df[s.column]):
            raise TypeError(
                f"pick='{s.label}' needs a numeric, boolean, datetime, timedelta or "
                f"ordered categorical column; got {df[s.column].dtype}. (Strings are "
                f"rejected on purpose: '10' < '9' lexicographically.)"
            )
    return selectors


def _rows_in(group: pd.DataFrame, result: Any) -> np.ndarray:
    """Positions within ``group`` of the rows a custom pick returned.

    Accepts a DataFrame subset, a single row (Series, e.g. ``g.loc[label]``),
    a boolean mask (Series aligned to the group, or array), or index labels.
    """
    if isinstance(result, pd.DataFrame):
        labels: Any = result.index
    elif isinstance(result, pd.Series) and result.dtype == bool and result.index.equals(group.index):
        return np.flatnonzero(result.to_numpy())
    elif isinstance(result, pd.Series):
        labels = [result.name]
    elif isinstance(result, np.ndarray) and result.dtype == bool and len(result) == len(group):
        return np.flatnonzero(result)
    else:
        labels = list(result) if pd.api.types.is_list_like(result) else [result]
    idx = group.index.get_indexer(labels)
    if (idx < 0).any():
        raise ValueError(
            f"custom pick returned rows that are not in its group: {list(np.asarray(labels, dtype=object)[idx < 0])[:5]}"
        )
    return np.unique(idx)


def _duplicated(codes: np.ndarray, keep: Any = "first") -> np.ndarray:
    return pd.Index(codes).duplicated(keep=keep)


def _select(
    frame: pd.DataFrame,
    codes: np.ndarray,
    selectors: List[_Selector],
    *,
    rng: Any,
    group_key: Callable[[int], Any],
) -> np.ndarray:
    """Run the selector chain; return sorted positions of the surviving rows."""
    pos = np.arange(len(frame))
    for i, sel in enumerate(selectors):
        c = codes[pos]
        if sel.kind in {"first", "last", "none"}:
            pos = pos[~_duplicated(c, False if sel.kind == "none" else sel.kind)]
        elif sel.kind == "random":
            perm = rng.permutation(len(pos))
            pos = np.sort(pos[perm[~_duplicated(c[perm])]])
        elif sel.kind in {"max", "min"}:
            v = frame[sel.column].iloc[pos].reset_index(drop=True)
            best = v.groupby(c).transform(sel.kind)
            hit = (v.eq(best) | (v.isna() & best.isna())).fillna(False).to_numpy(bool)
            pos = pos[hit]
            if i == len(selectors) - 1:  # final selector: break ties by input order
                pos = pos[~_duplicated(codes[pos])]
        else:
            # Sort once so each group is a cheap contiguous slice; codes are
            # numbered in first-seen order, so groups are visited in that order.
            order = pos[np.argsort(c, kind="stable")]
            ordered = frame.take(order)
            bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
            kept = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                group = ordered.iloc[a:b]
                kept.append(order[a:b][_rows_in(group, sel.fn(group, group_key(order[a])))])
            pos = np.sort(np.concatenate(kept)) if kept else pos[:0]
    return pos


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

_ALIASES = {
    "subset": "on", "cols": "on",
    "by": "group_by", "group": "group_by",
    "ids": "entity_by", "entity": "entity_by",
    "block": "block_by",
    "keep": "pick",
}

_ACTIONS = {"remove": "remove", "rm": "remove", "drop": "remove", "delete": "remove",
            "flag": "flag", "mark": "flag", "tag": "flag"}
_MODES = {"auto": "auto", "a": "auto", "collapse": "collapse", "c": "collapse",
          "one": "collapse", "within": "within", "w": "within"}
_SCOPES = {"global": "global", "per_column": "per_column", "per-column": "per_column",
           "column": "per_column"}


def _choice(value: Any, table: Dict[str, str], name: str) -> str:
    try:
        return table[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"{name} must be one of {sorted(set(table.values()))}; got {value!r}.") from None


def deduplicate_dataframe(
    df: pd.DataFrame,
    on: ColSpec = None,
    *,
    action: str = "remove",
    pick: Pick = _UNSET,
    group_by: ColSpec = None,
    entity_by: Union[ColSpec, bool] = None,
    block_by: ColSpec = None,
    mode: str = "auto",
    random_state: Any = None,
    flag_column: Hashable = "is_duplicate",
    group_id_column: Optional[Hashable] = None,
    return_stats: bool = False,
    log_stats: bool = True,
    entity_scope: str = "global",
    split_entity_values: bool = True,
    entity_separators_regex: Union[str, re.Pattern] = r"[;,|]+",
    entity_ignore_values: Optional[Iterable[str]] = None,
    coerce_token_types: bool = False,
    **aliases: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Deduplicate ``df``, optionally consolidating rows into groups or entities.

    Column specs (``on``, ``group_by``, ``entity_by``, ``block_by``) accept
    ``"a,b"``, ``"a b"``, ``"a|b"``, ``"*"``, or a list of exact labels (use a
    list for non-string labels). ``on`` also accepts exclusions (``"-ts"`` =
    all columns but ``ts``) and connectivity (``"any: email phone"``).

    Grouping
      - none:      exact duplicates on ``on`` (default: all columns).
      - group_by:  explicit groups.
      - entity_by: rows sharing ANY identifier token are one entity, linked
                   transitively (a-b via email, b-c via phone => a, b, c).
                   ``True``/``"auto"`` picks identity-like columns
                   conservatively. ``on="any: ..."`` is shorthand.
      - block_by:  hard partition applied to every strategy: rows in different
                   blocks are never grouped, linked or deduplicated together.
                   With both ``entity_by`` and ``group_by``, ``group_by`` joins
                   the blocks.

    mode
      - ``"auto"``: ``"collapse"`` when grouped, else ``"within"``.
      - ``"collapse"``: one survivor per group/entity.
      - ``"within"``: dedupe on ``on`` inside each group/entity.

    pick (who survives in each duplicate set)
      - ``"first"`` / ``"last"`` / ``"random"`` (seeded by ``random_state``:
        an int, ``None``, or a numpy Generator/RandomState).
      - ``"none"`` or ``False``: drop every member of a duplicate set (keeps
        singletons), like pandas ``keep=False``.
      - ``"max:col"`` / ``"min:col"`` (or ``("max", col)`` for non-string
        labels); ties go to the earliest row; an all-missing group keeps its
        first row.
      - ``callable(group, key)``: ``group`` is the group's rows (labelled by
        df's index, or by position if that index has duplicates); ``key`` is
        the group's key (scalar for one key column, else a tuple; the entity
        id in collapse mode). Return the rows to keep as a DataFrame subset,
        one row (Series), a boolean mask, or index labels. Selection only:
        returned values are not written back.
      - a list of the above, applied left to right on the shrinking survivor
        set: ``["max:score", "max:updated_at"]``. ``first``/``last``/
        ``random``/``none`` may only appear last.

    action
      - ``"remove"``: return the survivors, in input order, original index.
      - ``"flag"``: return every row with boolean ``flag_column`` set on
        non-survivors.

    group_id_column
      Optional output column with each row's group id: the entity/group id
      when grouped, otherwise the id of the row's duplicate set. Ids are
      dense integers numbered in order of first appearance.

    Entity tokens
      Strings are split on ``entity_separators_regex`` (when
      ``split_entity_values``), stripped and casefolded; lists, tuples, sets
      and numpy arrays are flattened. Tokens are type-tagged (``str:``,
      ``num:``, ``bool:``) so ``"123"`` never links to ``123`` unless
      ``coerce_token_types=True``; numbers compare by value (``123 == 123.0``).
      Placeholders in ``entity_ignore_values`` (default ``DEFAULT_IGNORE``,
      matched case-insensitively) never link. ``entity_scope="per_column"``
      links only same-column matches.

    Returns the output DataFrame, plus a stats dict if ``return_stats``.
    """
    start = time.perf_counter()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # ---- Aliases (subset=, by=, keep=, ...) ----
    params: Dict[str, Any] = {"on": on, "group_by": group_by, "entity_by": entity_by,
                              "block_by": block_by, "pick": pick}
    unknown = sorted(k for k in aliases if k not in _ALIASES)
    if unknown:
        raise TypeError(f"Unknown argument(s): {', '.join(unknown)}")
    for alias, value in aliases.items():
        target = _ALIASES[alias]
        if params[target] is not (_UNSET if target == "pick" else None):
            raise TypeError(f"Pass either {target}= or {alias}=, not both.")
        params[target] = value
    on, group_by, entity_by, block_by, pick = (
        params["on"], params["group_by"], params["entity_by"], params["block_by"], params["pick"])
    if pick is _UNSET:
        pick = "first"

    # ---- Options ----
    action_n = _choice(action, _ACTIONS, "action")
    mode_n = _choice(mode, _MODES, "mode")
    scope_n = _choice(entity_scope, _SCOPES, "entity_scope")
    if action_n == "flag" and flag_column in df.columns:
        raise ValueError(f"flag_column {flag_column!r} already exists in df.")
    if group_id_column is not None and group_id_column in df.columns:
        raise ValueError(f"group_id_column {group_id_column!r} already exists in df.")
    if action_n == "flag" and group_id_column == flag_column:
        raise ValueError("flag_column and group_id_column must differ.")

    # ---- Columns ----
    connect_any, on_cols = _parse_on(on, df)
    group_cols = _resolve_cols(group_by, df, name="group_by")
    block_cols = _resolve_cols(block_by, df, name="block_by") or []

    if entity_by is True or (isinstance(entity_by, str) and entity_by.strip().lower() == "auto"):
        entity_cols = _auto_entity_cols(df, exclude=block_cols + (group_cols or []))
        if not entity_cols:
            raise ValueError("entity_by='auto' found no identity-like columns; name them explicitly.")
    else:
        entity_cols = _resolve_cols(None if entity_by is False else entity_by, df, name="entity_by") or None
    if entity_cols is None and connect_any:
        entity_cols = on_cols
    if entity_cols is not None and group_cols:
        block_cols = list(dict.fromkeys(block_cols + group_cols))
        group_cols = None

    selectors = _parse_picks(pick, df)

    # ---- Group codes ----
    n = len(df)
    blocks = _key_codes(df, block_cols) if block_cols else None
    gid: Optional[np.ndarray] = None
    grouping = "none"
    if entity_cols:
        grouping = "entity_by"
        ignore = DEFAULT_IGNORE if entity_ignore_values is None else frozenset(
            str(v).strip().casefold() for v in entity_ignore_values)
        sep = re.compile(entity_separators_regex) if split_entity_values else None
        gid = _entity_codes(df, entity_cols, blocks=blocks, per_column=scope_n == "per_column",
                            sep=sep, ignore=ignore, coerce=coerce_token_types)
    elif group_cols:
        grouping = "group_by"
        gid = _key_codes(df, group_cols, base=blocks)

    if mode_n == "auto":
        mode_n = "collapse" if gid is not None else "within"

    if gid is not None and mode_n == "collapse":
        codes, key_cols = gid, []
    else:
        codes = _key_codes(df, on_cols, base=gid if gid is not None else blocks)
        key_cols = list(on_cols) if gid is not None else list(dict.fromkeys(block_cols + list(on_cols)))

    # ---- Survivors ----
    needs_labels = any(s.kind == "custom" for s in selectors) and not df.index.is_unique
    frame = df.set_axis(pd.RangeIndex(n), axis=0) if needs_labels else df
    rng = (random_state if isinstance(random_state, (np.random.Generator, np.random.RandomState))
           else np.random.default_rng(random_state))

    def group_key(p: int) -> Any:
        parts = ([int(gid[p])] if gid is not None else []) + _key_values(frame, key_cols, p)
        return parts[0] if len(parts) == 1 else tuple(parts)

    keep_pos = _select(frame, codes, selectors, rng=rng, group_key=group_key)
    out_ids = gid if gid is not None else codes

    # ---- Output ----
    if action_n == "remove":
        out = df.take(keep_pos)
        if group_id_column is not None:
            out[group_id_column] = out_ids[keep_pos]
    else:
        kept = np.zeros(n, dtype=bool)
        kept[keep_pos] = True
        out = df.copy()
        out[flag_column] = ~kept
        if group_id_column is not None:
            out[group_id_column] = out_ids

    dropped = n - len(keep_pos)
    stats: Dict[str, Any] = {
        "original_count": n,
        "kept_count": int(len(keep_pos)),
        "removed_count": dropped if action_n == "remove" else 0,
        "flagged_count": dropped if action_n == "flag" else 0,
        "group_count": int(out_ids.max()) + 1 if n else 0,
        "grouping": grouping,
        "mode": mode_n,
        "pick": [s.label for s in selectors] if isinstance(pick, list) else selectors[0].label,
        "on_kind": "any" if connect_any else "exact",
        "on_cols": list(on_cols),
        "entity_by": entity_cols if grouping == "entity_by" else None,
        "group_by": group_cols if grouping == "group_by" else None,
        "block_by": block_cols or None,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if log_stats:
        logger.info("Deduplication stats: %s", stats)
    return (out, stats) if return_stats else out


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame({
        "customer_id": [100, 100, 101, 102, 102, 103],
        "email": ["a@x.com", "a@x.com", "b@x.com", "c@x.com", None, "c@x.com"],
        "phone": ["555-1", None, "555-2", "555-3", "555-3", None],
        "score": [10, 12, 7, 20, 19, 18],
        "payload": ["v1", "v2", "v3", "v4", "v5", "v6"],
    })

    print(deduplicate_dataframe(df))                         # exact duplicates on all columns
    print(deduplicate_dataframe(df, on="email phone"))       # exact key (email, phone)
    print(deduplicate_dataframe(df, on="email", keep=False))  # drop every duplicated email
    print(deduplicate_dataframe(df, on="any: email phone", pick="max:score", group_id_column="entity_id"))
    print(deduplicate_dataframe(df, group_by="customer_id", pick=["max:score", "last"]))
    print(deduplicate_dataframe(df, group_by="customer_id", entity_by="email phone",
                                pick="max:score", group_id_column="entity_id"))
    print(deduplicate_dataframe(df, on="any: email phone", pick="max:score",
                                action="flag", group_id_column="entity_id"))
    print(deduplicate_dataframe(df, entity_by="auto", pick="max:score", group_id_column="entity_id"))
