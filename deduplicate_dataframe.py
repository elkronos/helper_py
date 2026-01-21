import functools
import inspect
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def profile(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Function %s executed in %.4f seconds", func.__name__, elapsed)
        return result

    return wrapper


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

Pick = Union[str, Callable[..., pd.DataFrame]]


@dataclass(frozen=True)
class PickSpec:
    kind: Literal["first", "last", "random", "max", "min", "custom"]
    column: Optional[str] = None


def _parse_pick(pick: Pick) -> Tuple[PickSpec, Optional[Callable[..., pd.DataFrame]]]:
    """
    pick can be:
      - "first" / "last" / "random"
      - "max:col" / "min:col"
      - callable: custom(group) or custom(group, key)
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
    sig = inspect.signature(custom_fn)
    if len(sig.parameters) >= 2:
        return custom_fn(group, getattr(group, "name", None))
    return custom_fn(group)


def _groupby_apply(grouped, func):
    """GroupBy.apply wrapper that uses modern behavior when available."""
    try:
        return grouped.apply(func, include_groups=False)
    except TypeError:
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


def _explode_id_value(
    v: Any,
    *,
    split: bool,
    sep_regex: str,
    strip: bool = True,
) -> List[Any]:
    """
    Turns a cell value into zero or more identifier tokens.
      - None/NaN -> []
      - list/tuple/set -> flattened recursively
      - string -> optionally split on separators (e.g., "a;b|c")
      - other scalar -> [value]
    """
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass

    if isinstance(v, (list, tuple, set, frozenset)):
        out: List[Any] = []
        for x in v:
            out.extend(_explode_id_value(x, split=split, sep_regex=sep_regex, strip=strip))
        return out

    if isinstance(v, str):
        s = v.strip() if strip else v
        if not s:
            return []
        if split and re.search(sep_regex, s):
            parts = [p.strip() for p in re.split(sep_regex, s) if p and p.strip()]
            return parts
        return [s]

    return [v]


def _group_ids_from_entity_by(
    df: pd.DataFrame,
    entity_cols: List[str],
    *,
    scope: Literal["global", "per_column"] = "global",
    split_values: bool = True,
    sep_regex: str = r"[;,|]+",
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
    idx = df.index.to_list()

    for i, row_index in enumerate(idx):
        for col in entity_cols:
            tokens = _explode_id_value(df.at[row_index, col], split=split_values, sep_regex=sep_regex)
            for t in tokens:
                key = (col, t) if scope == "per_column" else t
                j = first_seen.get(key)
                if j is None:
                    first_seen[key] = i
                else:
                    uf.union(i, j)

    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)
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


# ----------------------------
# Main API
# ----------------------------

@profile
def deduplicate_dataframe(
    df: pd.DataFrame,
    on: ColSpec = None,
    *,
    action: str = "remove",
    pick: Pick = "first",
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
      - callable(group)->DataFrame or callable(group, key)->DataFrame
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

    if "keep" in kwargs and pick == "first":
        pick = kwargs.pop("keep")

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unknown argument(s): {unknown}")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    # Normalize action
    a = str(action).strip().lower()
    if a in {"remove", "rm", "drop", "delete"}:
        action_n: Literal["remove", "flag"] = "remove"
    elif a in {"flag", "mark", "tag"}:
        action_n = "flag"
    else:
        raise ValueError("action must be 'remove'/'flag' (or a common synonym).")

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

    pick_spec, pick_fn = _parse_pick(pick)
    if pick_spec.kind in {"max", "min"}:
        assert pick_spec.column is not None
        if pick_spec.column not in df.columns:
            raise ValueError(f"pick column '{pick_spec.column}' not found in DataFrame.")

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
            )

    # Effective exact columns for "within" mode.
    # If on was connectivity-based, within mode uses the same column list as an exact key.
    on_cols_exact = on_spec.cols

    # ---- Compute rows to keep ----
    if group_id is None:
        # Global (no grouping): exact-key dedupe on on_cols_exact
        if pick_spec.kind in {"first", "last"}:
            keep = "first" if pick_spec.kind == "first" else "last"
            keep_df = work.drop_duplicates(subset=on_cols_exact, keep=keep)

        elif pick_spec.kind == "random":
            shuffled = work.sample(frac=1.0, random_state=random_state)
            keep_df = shuffled.drop_duplicates(subset=on_cols_exact, keep="first")
            keep_df = _reorder_like_input(work, keep_df)

        elif pick_spec.kind in {"max", "min"}:
            keep_df = _select_extreme_row(work, on_cols_exact, pick_spec.column, pick_spec.kind)  # type: ignore[arg-type]

        else:
            assert pick_fn is not None
            grouped = work.groupby(on_cols_exact, sort=False, group_keys=False)
            keep_df = _groupby_apply(grouped, lambda g: _call_custom(pick_fn, work.loc[g.index]))
            keep_df = _reorder_like_input(work, keep_df)

    else:
        # Grouped behavior
        gid_name = group_id.name or "_group_id"
        work2 = work.copy()
        work2[gid_name] = group_id

        if mode_n == "collapse":
            # One row per group/entity (consolidation)
            if pick_spec.kind == "first":
                keep_df = work2.groupby(gid_name, sort=False, group_keys=False).head(1)
            elif pick_spec.kind == "last":
                keep_df = work2.groupby(gid_name, sort=False, group_keys=False).tail(1)
            elif pick_spec.kind == "random":
                gb = work2.groupby(gid_name, sort=False, group_keys=False)
                try:
                    keep_df = gb.sample(n=1, random_state=random_state)
                except Exception:
                    keep_df = _groupby_apply(gb, lambda g: work2.loc[g.index].sample(n=1, random_state=random_state))
                keep_df = _reorder_like_input(work2, keep_df)
            elif pick_spec.kind in {"max", "min"}:
                keep_df = _select_extreme_row(work2, [gid_name], pick_spec.column, pick_spec.kind)  # type: ignore[arg-type]
            else:
                assert pick_fn is not None
                gb = work2.groupby(gid_name, sort=False, group_keys=False)
                keep_df = _groupby_apply(gb, lambda g: _call_custom(pick_fn, work2.loc[g.index]))
                keep_df = _reorder_like_input(work2, keep_df)

        else:
            # within: exact-key dedupe within each group/entity using on_cols_exact
            key_cols = _unique_preserve_order([gid_name] + list(on_cols_exact))

            if pick_spec.kind in {"first", "last"}:
                keep = "first" if pick_spec.kind == "first" else "last"
                keep_df = work2.drop_duplicates(subset=key_cols, keep=keep)

            elif pick_spec.kind == "random":
                gb = work2.groupby(gid_name, sort=False, group_keys=False)
                try:
                    shuffled = gb.sample(frac=1.0, random_state=random_state)
                except Exception:
                    shuffled = _groupby_apply(
                        gb, lambda g: work2.loc[g.index].sample(frac=1.0, random_state=random_state)
                    )
                keep_df = shuffled.drop_duplicates(subset=key_cols, keep="first")
                keep_df = _reorder_like_input(work2, keep_df)

            elif pick_spec.kind in {"max", "min"}:
                keep_df = _select_extreme_row(work2, key_cols, pick_spec.column, pick_spec.kind)  # type: ignore[arg-type]

            else:
                assert pick_fn is not None
                gb = work2.groupby(key_cols, sort=False, group_keys=False)
                keep_df = _groupby_apply(gb, lambda g: _call_custom(pick_fn, work2.loc[g.index]))
                keep_df = _reorder_like_input(work2, keep_df)

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

    # ---- Stats ----
    kept_count = int(len(keep_df))
    removed_count = int(original_count - kept_count)
    flagged_count = int(out[flag_column].sum()) if action_n == "flag" else 0

    stats = {
        "original_count": int(original_count),
        "kept_count": kept_count,
        "removed_count": removed_count if action_n == "remove" else 0,
        "flagged_count": flagged_count if action_n == "flag" else 0,
        "grouping": grouping_kind,
        "mode": mode_n,
        "pick": (pick if isinstance(pick, str) else getattr(pick, "__name__", "custom_callable")),
        "on_kind": on_spec.kind,
        "on_cols": on_spec.cols,
        "entity_by": entity_cols if grouping_kind == "entity_by" else None,
        "group_by": group_cols_for_grouping if grouping_kind == "group_by" else None,
        "block_by": block_cols if block_cols else None,
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
