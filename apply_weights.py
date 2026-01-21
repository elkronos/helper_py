import pandas as pd
import numpy as np
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union, Any, Dict


def apply_weights(
    data: pd.DataFrame,
    demographic_vars: Optional[Union[str, Sequence[str]]],
    demographic_weights: Mapping[str, Mapping[Any, float]],
    columns_to_weight: Optional[Union[str, Sequence[str], Callable[[pd.DataFrame], Sequence[str]]]] = None,
    new_column_names: Optional[Union[Sequence[str], Mapping[str, str], Callable[[str], str]]] = None,
    weight_suffix: str = "_weight",
    value_suffix: str = "_weighted",
    verbose: bool = False,
    handle_na: str = "drop",
    *,
    # Enhancements (all optional; defaults preserve existing behavior)
    default_weight: float = 1.0,
    require_all_levels: bool = True,          # if False, missing levels get default_weight
    validate_all_weights: bool = True,        # preserves your current validation behavior
    na_subset: Optional[Sequence[str]] = None,  # only used when handle_na == "drop"
    fill_values: Optional[Mapping[str, Any]] = None,  # per-column overrides when handle_na == "fill"
    total_weight_col: Optional[str] = None,   # if set, adds a column with the combined weight
    verbose_mode: str = "per_column",         # "per_column" (matches current), or "per_variable"
    copy: bool = True,
) -> pd.DataFrame:
    """
    Applies demographic weights to numeric columns in a DataFrame.

    Backward-compatible with the original function:
      - Same defaults for handle_na, naming, and verbose=True behavior (via verbose_mode="per_column")
      - Same strict validation by default (require_all_levels=True, validate_all_weights=True)

    Added flexibility:
      - demographic_vars and columns_to_weight can be a single string
      - columns_to_weight can be a callable (df -> list of columns)
      - new_column_names can be list, dict {orig: new}, or callable (orig -> new)
      - require_all_levels=False lets missing levels fall back to default_weight
      - na_subset controls dropna scope; fill_values provides targeted fills
      - total_weight_col exposes combined weights
      - verbose_mode="per_variable" avoids duplicating weight columns per value column
    """
    _check_input(
        data=data,
        demographic_vars=demographic_vars,
        demographic_weights=demographic_weights,
        columns_to_weight=columns_to_weight,
        new_column_names=new_column_names,
        handle_na=handle_na,
        default_weight=default_weight,
        require_all_levels=require_all_levels,
        validate_all_weights=validate_all_weights,
        verbose_mode=verbose_mode,
        na_subset=na_subset,
    )

    df = data.copy() if copy else data

    demographic_vars_list = _as_list_or_none(demographic_vars)
    if demographic_vars_list is None:
        # Intuitive shortcut: if not provided, use keys from demographic_weights (enhancement)
        demographic_vars_list = list(demographic_weights.keys())

    cols_to_weight = _resolve_columns_to_weight(df, columns_to_weight)

    # --- NaN handling (preserves existing defaults/behavior) ---
    if handle_na == "drop":
        df = df.dropna() if na_subset is None else df.dropna(subset=list(na_subset))
    elif handle_na == "fill":
        df = _fill_na(df, fill_values=fill_values)
    elif handle_na == "ignore":
        pass
    else:
        # should be caught by validation
        raise ValueError("handle_na must be 'drop', 'fill', or 'ignore'")

    weighted_df = df.copy()

    # --- Determine output column names (preserves existing behavior) ---
    out_names = _resolve_output_names(cols_to_weight, new_column_names, value_suffix)

    # --- Precompute weight series once (optimization) ---
    weight_series_by_var: Dict[str, pd.Series] = {}
    for var in demographic_vars_list:
        if var not in demographic_weights:
            continue  # preserves your current behavior: only apply if present in demographic_weights
        if var not in df.columns:
            raise KeyError(f"Demographic variable '{var}' not found in DataFrame columns")

        wmap = demographic_weights[var]

        # map -> NaN if missing; then fill to default_weight (matches original lambda get(..., 1.0))
        s = df[var].map(wmap).astype(float, errors="ignore")
        s = s.fillna(float(default_weight))

        if require_all_levels:
            # validate levels for requested vars only (unless validate_all_weights=True, handled in _check_input)
            missing_levels = set(pd.unique(df[var].dropna())) - set(wmap.keys())
            if missing_levels:
                raise ValueError(
                    f"All non-NaN levels of '{var}' must have a corresponding weight. Missing: {sorted(missing_levels)}"
                )

        weight_series_by_var[var] = s.astype(float)

    total_weights = pd.Series(1.0, index=df.index, dtype="float64")
    for s in weight_series_by_var.values():
        total_weights = total_weights.mul(s, fill_value=1.0)

    if total_weight_col:
        weighted_df[total_weight_col] = total_weights

    # --- Verbose weight outputs ---
    if verbose:
        if verbose_mode not in {"per_column", "per_variable"}:
            raise ValueError("verbose_mode must be 'per_column' or 'per_variable'")

        if verbose_mode == "per_variable":
            # compact: one weight column per demographic variable
            for var, s in weight_series_by_var.items():
                weighted_df[f"{var}{weight_suffix}"] = s
        else:
            # preserves your current behavior: repeat weights per (value column, demographic var)
            # (still cheaper because we reuse the same s)
            pass

    # --- Apply weights to each requested numeric column ---
    for col, out_col in zip(cols_to_weight, out_names):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric to apply weights")

        if verbose and verbose_mode == "per_column":
            for var, s in weight_series_by_var.items():
                weighted_df[f"{col}_{var}{weight_suffix}"] = s

        weighted_df[out_col] = df[col].mul(total_weights)

    return weighted_df


# ------------------------- helpers -------------------------

def _as_list_or_none(x: Optional[Union[str, Sequence[str]]]) -> Optional[list]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _resolve_columns_to_weight(
    df: pd.DataFrame,
    columns_to_weight: Optional[Union[str, Sequence[str], Callable[[pd.DataFrame], Sequence[str]]]],
) -> list:
    if columns_to_weight is None:
        return df.select_dtypes(include=["number"]).columns.tolist()
    if callable(columns_to_weight):
        cols = list(columns_to_weight(df))
        return cols
    if isinstance(columns_to_weight, str):
        return [columns_to_weight]
    return list(columns_to_weight)


def _resolve_output_names(
    cols_to_weight: Sequence[str],
    new_column_names: Optional[Union[Sequence[str], Mapping[str, str], Callable[[str], str]]],
    value_suffix: str,
) -> list:
    if new_column_names is None:
        return [f"{c}{value_suffix}" for c in cols_to_weight]

    if callable(new_column_names):
        return [str(new_column_names(c)) for c in cols_to_weight]

    if isinstance(new_column_names, Mapping):
        return [str(new_column_names.get(c, f"{c}{value_suffix}")) for c in cols_to_weight]

    # sequence
    new_column_names = list(new_column_names)
    if len(new_column_names) != len(cols_to_weight):
        raise ValueError("Length of new_column_names must equal length of columns_to_weight")
    return [str(x) for x in new_column_names]


def _fill_na(df: pd.DataFrame, fill_values: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    fill_values = dict(fill_values or {})
    out = df.copy()

    for col in out.columns:
        if col in fill_values:
            out[col] = out[col].fillna(fill_values[col])
            continue

        if pd.api.types.is_numeric_dtype(out[col]):
            mean_val = out[col].mean()
            out[col] = out[col].fillna(mean_val)
        else:
            mode = out[col].mode(dropna=True)
            if len(mode) > 0:
                out[col] = out[col].fillna(mode.iloc[0])
            # else: leave as NaN (safer than inventing a label)
    return out


def _check_input(
    data: pd.DataFrame,
    demographic_vars: Optional[Union[str, Sequence[str]]],
    demographic_weights: Mapping[str, Mapping[Any, float]],
    columns_to_weight: Optional[Union[str, Sequence[str], Callable[[pd.DataFrame], Sequence[str]]]],
    new_column_names: Optional[Union[Sequence[str], Mapping[str, str], Callable[[str], str]]],
    handle_na: str,
    default_weight: float,
    require_all_levels: bool,
    validate_all_weights: bool,
    verbose_mode: str,
    na_subset: Optional[Sequence[str]],
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")

    if demographic_vars is not None:
        if isinstance(demographic_vars, str):
            pass
        elif not isinstance(demographic_vars, (list, tuple)):
            raise TypeError("demographic_vars must be a string, a list/tuple of strings, or None")
        else:
            if not all(isinstance(v, str) for v in demographic_vars):
                raise TypeError("demographic_vars must contain only strings")

    if not isinstance(demographic_weights, Mapping) or not all(
        isinstance(w, Mapping) for w in demographic_weights.values()
    ):
        raise TypeError("demographic_weights must be a mapping of {var: {level: weight}}")

    # Ensure weights numeric
    for var, wmap in demographic_weights.items():
        for level, w in wmap.items():
            if not isinstance(w, (int, float, np.integer, np.floating)):
                raise TypeError(f"All weights for '{var}' must be numeric; got {type(w)} for level '{level}'")

    if columns_to_weight is not None and not (
        isinstance(columns_to_weight, str)
        or callable(columns_to_weight)
        or isinstance(columns_to_weight, (list, tuple))
    ):
        raise TypeError("columns_to_weight must be None, a string, a list/tuple of strings, or a callable(df)->list")

    if new_column_names is not None and not (
        callable(new_column_names) or isinstance(new_column_names, Mapping) or isinstance(new_column_names, (list, tuple))
    ):
        raise TypeError("new_column_names must be None, a list/tuple, a dict mapping, or a callable(col)->name")

    if handle_na not in {"drop", "fill", "ignore"}:
        raise ValueError("handle_na must be 'drop', 'fill', or 'ignore'")

    if not isinstance(default_weight, (int, float, np.integer, np.floating)):
        raise TypeError("default_weight must be numeric")

    if not isinstance(require_all_levels, bool):
        raise TypeError("require_all_levels must be bool")

    if not isinstance(validate_all_weights, bool):
        raise TypeError("validate_all_weights must be bool")

    if verbose_mode not in {"per_column", "per_variable"}:
        raise ValueError("verbose_mode must be 'per_column' or 'per_variable'")

    if na_subset is not None:
        if not isinstance(na_subset, (list, tuple)) or not all(isinstance(c, str) for c in na_subset):
            raise TypeError("na_subset must be None or a list/tuple of column names")

    # Preserve your original behavior by default: validate weights for *all* weight vars present in df
    if validate_all_weights:
        for var, wmap in demographic_weights.items():
            if var in data.columns:
                unique_levels = pd.unique(data[var].dropna())
                if require_all_levels and not all(level in wmap for level in unique_levels):
                    missing = sorted(set(unique_levels) - set(wmap.keys()))
                    raise ValueError(f"All non-NaN levels of '{var}' must have a corresponding weight. Missing: {missing}")


'''
# ------------------------- example (your existing tests still work) -------------------------
if __name__ == "__main__":
    data_with_nan = pd.DataFrame({
        "income": [np.nan, 60000, 75000, 80000, 90000, 100000],
        "age": [25, 30, 35, 40, 45, 50],
        "gender": ["male", "female", "male", "female", "male", "female"],
        "education": ["high school", "bachelor", np.nan, "bachelor", "phd", "master"],
    })

    demographic_weights = {
        "gender": {"male": 0.8, "female": 1.2},
        "education": {"high school": 1.5, "bachelor": 1.0, "master": 0.8, "phd": 0.7},
    }

    result_drop = apply_weights(
        data_with_nan,
        ["gender", "education"],
        demographic_weights,
        handle_na="drop",
        verbose=True,
    )

    result_fill = apply_weights(
        data_with_nan,
        ["gender", "education"],
        demographic_weights,
        handle_na="fill",
        verbose=True,
    )

    result_ignore = apply_weights(
        data_with_nan,
        ["gender", "education"],
        demographic_weights,
        handle_na="ignore",
        verbose=True,
    )

    # New (optional) conveniences:
    # - demographic_vars=None -> use keys from demographic_weights
    # - total_weight_col to expose combined weights
    # - require_all_levels=False to allow unseen levels
    # - verbose_mode="per_variable" to avoid repeated weight columns
    result_compact = apply_weights(
        data_with_nan,
        None,
        demographic_weights,
        columns_to_weight="income",
        handle_na="fill",
        verbose=True,
        verbose_mode="per_variable",
        total_weight_col="weight_total",
        require_all_levels=False,
    )

    print(result_drop)
    print(result_fill)
    print(result_ignore)
    print(result_compact)

'''