#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
appropriateness_evaluator

Given a context and a dataset of example "appropriate" lists (e.g., recipes),
decide whether a new item (e.g., ingredient) is appropriate.

Signals:
- Embedding similarity to known items (nearest neighbors)
- Frequency of appearance across (non-empty) rows (prior)
- Optional embedding outlier detection (LOF)
- Optional LLM veto (context-aware)

Security note:
Pickle files can execute code during load; only load pickle files from trusted sources.
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import pickle
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd

_SENTENCE_TRANSFORMERS_IMPORT_ERROR: Optional[BaseException] = None
_SKLEARN_IMPORT_ERROR: Optional[BaseException] = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = e

try:
    from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor  # type: ignore
except Exception as e:  # pragma: no cover
    NearestNeighbors = None  # type: ignore
    LocalOutlierFactor = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = e


LOG = logging.getLogger("appropriateness")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


_SPLIT_RE = re.compile(r"[,\n;|]+|\s+/\s+")

_UNIT_TOKENS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "oz", "ounce", "ounces", "g", "gram", "grams", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters", "l", "liter", "liters",
    "lb", "lbs", "pound", "pounds",
    "pinch", "pinches", "dash", "dashes",
    "quart", "quarts", "qt", "qts", "pint", "pints",
    "slice", "slices", "clove", "cloves", "can", "cans", "package", "packages",
    "inch", "inches", "cm", "centimeter", "centimeters", "mm", "millimeter", "millimeters",
    "meter", "meters",
    "stick", "sticks", "bunch", "bunches", "sprig", "sprigs", "piece", "pieces",
    "handful", "handfuls", "drop", "drops", "sheet", "sheets",
}

_UNICODE_FRACTIONS = set("¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞")
_UNICODE_FRACTIONS_ESC = re.escape("".join(_UNICODE_FRACTIONS))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_item(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""

    s = re.sub(r"(?<!\w)\d+\s*-\s*\d+\s*/\s*\d+(?!\w)", " ", s)  # 1-1/2
    s = re.sub(r"(?<!\w)\d+\s+\d+\s*/\s*\d+(?!\w)", " ", s)      # 1 1/2
    s = re.sub(r"(?<!\w)\d+\s*/\s*\d+(?!\w)", " ", s)            # 1/2
    s = re.sub(r"(?<!\w)\d+\s*/\s*\d+(?=\w)", " ", s)            # 1/2cup

    s = re.sub(r"(?<!\w)\d+\s*(?:to|[-–—])\s*\d+(?!\w)", " ", s)  # 2-3
    s = re.sub(r"(?<!\w)\d+\s*(?:to|[-–—])\s*\d+(?=\w)", " ", s)  # 2-3inch

    s = re.sub(rf"(?<!\w)\d*[{_UNICODE_FRACTIONS_ESC}]+(?!\w)", " ", s)  # ½
    s = re.sub(rf"(?<!\w)\d*[{_UNICODE_FRACTIONS_ESC}]+(?=\w)", " ", s)  # ½cup

    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    toks_in = s.split()
    toks_out: List[str] = []

    has_non_unit_alpha = any(any(ch.isalpha() for ch in t) and (t not in _UNIT_TOKENS) for t in toks_in)
    any_alpha_in = any(any(ch.isalpha() for ch in t) for t in toks_in)

    for i, t in enumerate(toks_in):
        if has_non_unit_alpha and t in _UNIT_TOKENS:
            continue

        if any(ch.isalpha() for ch in t):
            toks_out.append(t)
            continue

        is_small_int = t.isdigit() and len(t) <= 2
        if not is_small_int:
            continue

        prev_tok = toks_in[i - 1] if i - 1 >= 0 else ""
        next_tok = toks_in[i + 1] if i + 1 < len(toks_in) else ""
        if prev_tok in _UNIT_TOKENS or next_tok in _UNIT_TOKENS:
            continue

        if not any_alpha_in:
            continue

        toks_out.append(t)

    out = " ".join(toks_out).strip()
    if not out:
        return ""

    out_toks = out.split()
    any_alpha_non_unit = any(any(ch.isalpha() for ch in tok) and tok not in _UNIT_TOKENS for tok in out_toks)
    any_alpha = any(any(ch.isalpha() for ch in tok) for tok in out_toks)
    all_units_or_nums = all((tok in _UNIT_TOKENS) or tok.isdigit() for tok in out_toks)

    if (not any_alpha_non_unit) and all_units_or_nums:
        return ""
    if any_alpha and not any_alpha_non_unit and all(tok in _UNIT_TOKENS for tok in out_toks):
        return ""

    return out


def split_items(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = _SPLIT_RE.split(str(raw))
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _is_null_like(value: Any) -> bool:
    try:
        res = pd.isna(value)
        if isinstance(res, (bool, np.bool_)):
            return bool(res)
        arr = np.asarray(res)
        if arr.size == 0:
            return True
        return bool(np.all(arr))
    except Exception:
        return False


def _iterable_scalars(value: Any) -> Optional[Iterable[Any]]:
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return value
    return None


def extract_items_from_row(value: Any) -> List[str]:
    if value is None or _is_null_like(value):
        return []

    parts: List[str] = []
    it = _iterable_scalars(value)

    if it is not None and not isinstance(value, (str, bytes)):
        for x in it:
            if x is None or _is_null_like(x):
                continue
            if isinstance(x, str):
                parts.extend(split_items(x))
            else:
                parts.append(str(x))
    else:
        parts = split_items(str(value))

    cleaned = [normalize_item(p) for p in parts]
    return [c for c in cleaned if c]


@dataclass
class Evaluation:
    item: str
    normalized: str
    decision: str
    score: float
    best_match: Optional[str] = None
    best_similarity: Optional[float] = None
    best_freq: Optional[float] = None
    outlier: Optional[bool] = None
    llm_p: Optional[float] = None
    reason: Optional[str] = None
    neighbors: Optional[List[Tuple[str, float, float]]] = None


@dataclass(frozen=True)
class Thresholds:
    min_sim: float
    min_freq: float
    allow_outlier: bool
    llm_enabled: bool
    llm_min_p: float


PRESETS: Dict[str, Thresholds] = {
    "strict": Thresholds(min_sim=0.62, min_freq=0.08, allow_outlier=False, llm_enabled=False, llm_min_p=0.40),
    "normal": Thresholds(min_sim=0.56, min_freq=0.04, allow_outlier=False, llm_enabled=False, llm_min_p=0.35),
    "lenient": Thresholds(min_sim=0.52, min_freq=0.01, allow_outlier=True,  llm_enabled=False, llm_min_p=0.30),
}


class LLMVeto:
    """
    Best-effort OpenAI client wrapper.
    If calls fail at runtime, veto is disabled for the remainder of the run to avoid silent, repeated failures.
    """

    def __init__(self, api_key: Optional[str], model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        self._client = None
        self._legacy = None
        self._mode: Optional[str] = None
        self._disabled_reason: Optional[str] = None

        if not self.api_key:
            self._disabled_reason = "OPENAI_API_KEY is not set"
            return

        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)
            self._mode = "v1"
            return
        except Exception as e:
            self._client = None
            self._disabled_reason = f"OpenAI v1 client init failed: {e!r}"

        try:
            import openai  # type: ignore
            openai.api_key = self.api_key
            self._legacy = openai
            self._mode = "legacy"
            self._disabled_reason = None
        except Exception as e:
            self._legacy = None
            self._mode = None
            self._disabled_reason = f"OpenAI legacy client init failed: {e!r}"

    @property
    def available(self) -> bool:
        return bool(self.api_key and (self._mode in {"v1", "legacy"}))

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"\s*```$", "", t).strip()
        return t

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        t = self._strip_code_fences(text)
        if not t:
            return None

        start = t.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i + 1].strip()

        return None

    def _disable(self, reason: str) -> None:
        self._mode = None
        self._client = None
        self._legacy = None
        self._disabled_reason = reason

    def score(self, context: str, item: str, timeout_s: float = 20.0) -> Optional[Tuple[float, str]]:
        if not self.available:
            return None

        prompt = (
            "You are judging whether an item is appropriate for a context.\n\n"
            f"Context:\n{context}\n\n"
            f"Item:\n{item}\n\n"
            "Return ONLY JSON with this schema:\n"
            '{ "p": number, "reason": string }\n'
            "Where p is the probability (0 to 1) the item is appropriate in context.\n"
        )

        try:
            if self._mode == "v1" and self._client is not None:
                kwargs = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=180,
                )
                try:
                    resp = self._client.chat.completions.create(**kwargs, timeout=timeout_s)
                except TypeError:
                    resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()

            elif self._mode == "legacy" and self._legacy is not None:
                resp = self._legacy.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=180,
                    request_timeout=timeout_s,
                )
                text = (resp["choices"][0]["message"]["content"] or "").strip()
            else:
                return None

        except Exception as e:
            LOG.warning("LLM veto call failed; disabling veto for remainder of run (%s)", e)
            self._disable(f"LLM call failed: {e!r}")
            return None

        try:
            candidate = self._extract_first_json_object(text)
            if not candidate:
                candidate = self._strip_code_fences(text)
            if not candidate:
                raise ValueError("empty response")
            obj = json.loads(candidate)
            p = float(obj.get("p"))
            reason = str(obj.get("reason", "")).strip()
            return (clamp(p, 0.0, 1.0), reason)
        except Exception:
            m = re.search(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
            if not m:
                return None
            p = float(m.group(1))
            return (clamp(p, 0.0, 1.0), "LLM parsing fallback")


@dataclass(frozen=True)
class IndexStats:
    n_rows_total: int
    n_rows_used: int
    n_unique: int
    suggested_min_sim: float
    suggested_min_freq: float


class IngredientIndex:
    def __init__(
        self,
        items: List[str],
        freqs: Union[np.ndarray, List[float]],
        embed_model_name: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 10,
        enable_lof: bool = True,
        lof_contamination: float = 0.08,
    ):
        if NearestNeighbors is None:  # pragma: no cover
            raise RuntimeError(
                "scikit-learn is required but could not be imported. "
                "Install with: pip install scikit-learn\n"
                f"Import error: {_SKLEARN_IMPORT_ERROR!r}"
            )

        self.items = list(items)
        self.freqs = np.asarray(freqs, dtype=np.float32)

        if len(self.items) != len(self.freqs):
            raise ValueError(f"freqs length mismatch: len(items)={len(self.items)} != len(freqs)={len(self.freqs)}")
        if len(self.items) == 0:
            raise ValueError("Cannot create index with zero items.")
        if len(set(self.items)) != len(self.items):
            raise ValueError("Items must be unique.")
        if not np.all(np.isfinite(self.freqs)):
            raise ValueError("freqs contains NaN/Inf.")
        if np.any(self.freqs < 0.0) or np.any(self.freqs > 1.0):
            raise ValueError("freqs must be in [0, 1].")

        self.embed_model_name = embed_model_name
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be >= 2.")
        self.enable_lof = bool(enable_lof)
        self.lof_contamination = float(lof_contamination)

        if self.enable_lof and LocalOutlierFactor is None:  # pragma: no cover
            raise RuntimeError(
                "LocalOutlierFactor requested but scikit-learn LOF could not be imported. "
                "Install with: pip install scikit-learn\n"
                f"Import error: {_SKLEARN_IMPORT_ERROR!r}"
            )

        self._model: Optional[Any] = None
        self.embeddings: Optional[np.ndarray] = None
        self.nn: Optional[Any] = None
        self.lof: Optional[Any] = None

        self._item_to_idx = {it: i for i, it in enumerate(self.items)}

    @classmethod
    def from_dataset(
        cls,
        df: pd.DataFrame,
        column: str,
        embed_model_name: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 10,
        enable_lof: bool = True,
        lof_contamination: float = 0.08,
    ) -> Tuple["IngredientIndex", IndexStats]:
        if column not in df.columns:
            raise ValueError(f"Dataset missing column '{column}'. Found: {list(df.columns)}")

        counts: Dict[str, int] = {}
        n_rows_total = 0
        n_rows_used = 0

        for v in df[column].tolist():
            n_rows_total += 1
            row_items = set(extract_items_from_row(v))
            if not row_items:
                continue
            n_rows_used += 1
            for it in row_items:
                counts[it] = counts.get(it, 0) + 1

        if n_rows_total == 0:
            raise ValueError("Dataset has zero rows.")
        if n_rows_used == 0:
            raise ValueError("Dataset has no usable (non-empty) rows after normalization.")

        items = sorted(counts.keys())
        if len(items) < 2:
            raise ValueError(
                f"Not enough unique items to build an index (n_unique={len(items)}). "
                "Need at least 2 unique items."
            )

        freqs = np.array([counts[it] / n_rows_used for it in items], dtype=np.float32)

        idx = cls(
            items=items,
            freqs=freqs,
            embed_model_name=embed_model_name,
            n_neighbors=n_neighbors,
            enable_lof=enable_lof,
            lof_contamination=lof_contamination,
        )
        idx.fit()
        stats = idx._compute_stats(n_rows_total=n_rows_total, n_rows_used=n_rows_used)
        return idx, stats

    def fit(
        self,
        precomputed_embeddings: Optional[np.ndarray] = None,
        allow_missing_embedder: bool = False,
    ) -> None:
        if len(self.items) < 2:
            raise ValueError(f"Cannot fit index with fewer than 2 items (n={len(self.items)}).")

        self._model = None

        if SentenceTransformer is None:  # pragma: no cover
            if not allow_missing_embedder:
                raise RuntimeError(
                    "sentence-transformers is required but could not be imported. "
                    "Install with: pip install sentence-transformers\n"
                    f"Import error: {_SENTENCE_TRANSFORMERS_IMPORT_ERROR!r}"
                )
            LOG.warning(
                "sentence-transformers is not available; non-exact evaluation will be disabled (%r).",
                _SENTENCE_TRANSFORMERS_IMPORT_ERROR,
            )
        else:
            try:
                LOG.info("Loading embedding model: %s", self.embed_model_name)
                self._model = SentenceTransformer(self.embed_model_name)
            except Exception as e:
                if not allow_missing_embedder:
                    raise RuntimeError(
                        f"Failed to load embedding model '{self.embed_model_name}'. "
                        "Check model name and environment/network.\n"
                        f"Root error: {e!r}"
                    ) from e
                self._model = None
                LOG.warning(
                    "Failed to load embedding model '%s'; non-exact evaluation will be disabled (%r).",
                    self.embed_model_name,
                    e,
                )

        if precomputed_embeddings is not None:
            emb = np.asarray(precomputed_embeddings, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != len(self.items):
                raise ValueError(
                    f"Invalid precomputed embeddings shape: {emb.shape}. "
                    f"Expected (n_items, dim)=({len(self.items)}, dim)."
                )
            if not np.all(np.isfinite(emb)):
                raise ValueError("Embeddings contain NaN/Inf.")
            self.embeddings = emb
        else:
            if self._model is None:
                raise RuntimeError("Embedding model is not available; cannot compute embeddings.")
            try:
                LOG.info("Embedding %d unique items...", len(self.items))
                emb = np.asarray(
                    self._model.encode(self.items, convert_to_tensor=False),
                    dtype=np.float32,
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed during embedding computation. This is often caused by an incompatible "
                    "transformers/torch stack or missing model files.\n"
                    f"Root error: {e!r}"
                ) from e

            if emb.ndim != 2 or emb.shape[0] != len(self.items):
                raise ValueError("Model returned embeddings with an unexpected shape.")
            if not np.all(np.isfinite(emb)):
                raise ValueError("Embeddings contain NaN/Inf.")
            self.embeddings = emb

        k = min(max(2, self.n_neighbors), len(self.items))
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.nn.fit(self.embeddings)

        self.lof = None
        if self.enable_lof and LocalOutlierFactor is not None and len(self.items) >= 40:
            try:
                nnb = min(20, max(5, len(self.items) // 20))
                self.lof = LocalOutlierFactor(
                    n_neighbors=min(nnb, len(self.items) - 1),
                    contamination=clamp(self.lof_contamination, 0.01, 0.25),
                    novelty=True,
                    metric="cosine",
                )
                self.lof.fit(self.embeddings)
            except Exception as e:
                LOG.debug("LOF disabled due to initialization/fit error: %s", e)
                self.lof = None

    def _compute_stats(self, n_rows_total: int, n_rows_used: int) -> IndexStats:
        suggested_min_freq = clamp(2.0 / max(1, n_rows_used), 0.01, 0.10)

        suggested_min_sim = 0.56
        try:
            if self.nn is not None and self.embeddings is not None and len(self.items) >= 10:
                dists, _ = self.nn.kneighbors(self.embeddings, n_neighbors=2, return_distance=True)
                nn_sim = 1.0 - dists[:, 1]
                suggested_min_sim = float(np.percentile(nn_sim, 5))
                suggested_min_sim = clamp(suggested_min_sim, 0.45, 0.75)
        except Exception:
            pass

        return IndexStats(
            n_rows_total=int(n_rows_total),
            n_rows_used=int(n_rows_used),
            n_unique=int(len(self.items)),
            suggested_min_sim=float(suggested_min_sim),
            suggested_min_freq=float(suggested_min_freq),
        )

    def has_exact(self, normalized_item: str) -> bool:
        return normalized_item in self._item_to_idx

    def freq_of(self, normalized_item: str) -> Optional[float]:
        ix = self._item_to_idx.get(normalized_item)
        if ix is None:
            return None
        return float(self.freqs[ix])

    def embedding_of_known_item(self, normalized_item: str) -> Optional[np.ndarray]:
        ix = self._item_to_idx.get(normalized_item)
        if ix is None or self.embeddings is None:
            return None
        return np.asarray(self.embeddings[ix], dtype=np.float32)

    def embed_one(self, normalized_item: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedding model is not available; cannot embed new items.")
        emb = np.asarray(self._model.encode([normalized_item], convert_to_tensor=False), dtype=np.float32)[0]
        if not np.all(np.isfinite(emb)):
            raise ValueError("Embedding contains NaN/Inf.")
        return emb

    def neighbors(
        self,
        emb: np.ndarray,
        k: int = 5,
        exclude_item: Optional[str] = None,
    ) -> List[Tuple[str, float, float]]:
        if self.nn is None:
            return []
        k = max(1, min(int(k), len(self.items)))

        request_k = k
        if exclude_item is not None and len(self.items) > k:
            request_k = min(len(self.items), k + 1)

        dists, idxs = self.nn.kneighbors([emb], n_neighbors=request_k, return_distance=True)

        out: List[Tuple[str, float, float]] = []
        for dist, ix in zip(dists[0], idxs[0]):
            it = self.items[int(ix)]
            if exclude_item is not None and it == exclude_item:
                continue
            sim = float(1.0 - float(dist))
            freq = float(self.freqs[int(ix)])
            out.append((it, sim, freq))
            if len(out) >= k:
                break
        return out

    def is_outlier(self, emb: np.ndarray) -> Optional[bool]:
        if self.lof is None:
            return None
        try:
            pred = self.lof.predict([emb])[0]
            return bool(pred == -1)
        except Exception:
            return None


class AppropriatenessEvaluator:
    def __init__(
        self,
        context: str,
        index: IngredientIndex,
        thresholds: Thresholds,
        k_neighbors: int = 5,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        llm_veto: Optional[LLMVeto] = None,
    ):
        self.context = (context or "").strip()
        self.index = index
        self.thresholds = thresholds
        self.k_neighbors = int(k_neighbors)
        self.llm_veto = llm_veto if llm_veto is not None else LLMVeto(api_key=api_key, model=llm_model)

        if self.thresholds.llm_enabled and not self.llm_veto.available:
            reason = self.llm_veto.disabled_reason or "unknown"
            LOG.warning("LLM veto was requested but is not available; proceeding without it (%s).", reason)
            self.thresholds = Thresholds(
                min_sim=self.thresholds.min_sim,
                min_freq=self.thresholds.min_freq,
                allow_outlier=self.thresholds.allow_outlier,
                llm_enabled=False,
                llm_min_p=self.thresholds.llm_min_p,
            )

    def _freq_bonus(self, freq: float) -> float:
        return clamp(math.log1p(freq * 10.0) / math.log1p(10.0), 0.0, 1.0)

    def _appropriateness_score(
        self,
        sim: float,
        freq: float,
        min_freq: float,
        outlier: Optional[bool],
        allow_outlier: bool,
    ) -> float:
        sim = clamp(sim, 0.0, 1.0)
        freq = clamp(freq, 0.0, 1.0)

        base = clamp(0.80 * sim + 0.20 * self._freq_bonus(freq), 0.0, 1.0)

        if min_freq > 0 and freq < min_freq:
            rarity_factor = clamp(freq / min_freq, 0.0, 1.0)
            base *= 0.25 + 0.75 * rarity_factor

        if outlier is True and not allow_outlier:
            base *= 0.7

        return clamp(base, 0.0, 1.0)

    def evaluate(self, item: str, explain: bool = False) -> Evaluation:
        raw = item
        norm = normalize_item(item)

        if not norm:
            return Evaluation(
                item=raw,
                normalized=norm,
                decision="Inappropriate",
                score=0.0,
                reason="empty/invalid input",
            )

        if self.index.has_exact(norm):
            freq_val = self.index.freq_of(norm)
            freq = float(freq_val) if freq_val is not None else 0.0

            emb_known = self.index.embedding_of_known_item(norm)
            outlier: Optional[bool] = None
            neighbors: Optional[List[Tuple[str, float, float]]] = None

            if emb_known is not None:
                outlier = self.index.is_outlier(emb_known)
                if explain:
                    neighbors = self.index.neighbors(
                        emb_known,
                        k=self.k_neighbors,
                        exclude_item=norm,
                    )

            sim = 1.0
            score = self._appropriateness_score(
                sim=sim,
                freq=freq,
                min_freq=self.thresholds.min_freq,
                outlier=outlier,
                allow_outlier=self.thresholds.allow_outlier,
            )

            if freq < self.thresholds.min_freq:
                decision = "Inappropriate"
                reason = "exact match; too uncommon"
            elif outlier is True and not self.thresholds.allow_outlier:
                decision = "Inappropriate"
                reason = "embedding outlier vs reference set (exact match)"
            else:
                decision = "Appropriate"
                reason = "exact match; passes thresholds"

            llm_p = None
            if self.thresholds.llm_enabled and decision == "Appropriate":
                llm = self.llm_veto.score(self.context, raw)
                if llm is not None:
                    llm_p, llm_reason = llm
                    if llm_p < self.thresholds.llm_min_p:
                        score = min(score, llm_p)
                        return Evaluation(
                            item=raw,
                            normalized=norm,
                            decision="Inappropriate",
                            score=score,
                            best_match=norm,
                            best_similarity=1.0,
                            best_freq=freq,
                            outlier=outlier,
                            llm_p=llm_p,
                            reason=f"LLM veto: {llm_reason}",
                            neighbors=neighbors,
                        )

            return Evaluation(
                item=raw,
                normalized=norm,
                decision=decision,
                score=score,
                best_match=norm,
                best_similarity=1.0,
                best_freq=freq,
                outlier=outlier,
                llm_p=llm_p,
                reason=reason,
                neighbors=neighbors,
            )

        try:
            emb = self.index.embed_one(norm)
        except Exception:
            return Evaluation(
                item=raw,
                normalized=norm,
                decision="Inappropriate",
                score=0.0,
                reason="embedding model unavailable; cannot evaluate non-exact item",
            )

        nbrs = self.index.neighbors(emb, k=max(self.k_neighbors, 5))
        if not nbrs:
            return Evaluation(item=raw, normalized=norm, decision="Inappropriate", score=0.0, reason="no reference data")

        best_item, best_sim, best_freq = nbrs[0]
        outlier = self.index.is_outlier(emb)

        score = self._appropriateness_score(
            sim=best_sim,
            freq=best_freq,
            min_freq=self.thresholds.min_freq,
            outlier=outlier,
            allow_outlier=self.thresholds.allow_outlier,
        )

        if best_sim < self.thresholds.min_sim:
            decision = "Inappropriate"
            reason = f"low similarity to known items (best={best_item}, sim={best_sim:.3f})"
        elif best_freq < self.thresholds.min_freq:
            decision = "Inappropriate"
            reason = f"too uncommon in dataset (best={best_item}, freq={best_freq:.3f})"
        elif outlier is True and not self.thresholds.allow_outlier:
            decision = "Inappropriate"
            reason = "embedding outlier vs reference set"
        else:
            decision = "Appropriate"
            reason = f"similar to '{best_item}' (sim={best_sim:.3f}, freq={best_freq:.3f})"

        llm_p = None
        if decision == "Appropriate" and self.thresholds.llm_enabled:
            llm = self.llm_veto.score(self.context, raw)
            if llm is not None:
                llm_p, llm_reason = llm
                if llm_p < self.thresholds.llm_min_p:
                    decision = "Inappropriate"
                    reason = f"LLM veto: {llm_reason}"
                    score = min(score, llm_p)

        return Evaluation(
            item=raw,
            normalized=norm,
            decision=decision,
            score=score,
            best_match=best_item,
            best_similarity=best_sim,
            best_freq=best_freq,
            outlier=outlier,
            llm_p=llm_p,
            reason=reason,
            neighbors=nbrs[: self.k_neighbors] if explain else None,
        )


@dataclass(frozen=True)
class SavedModel:
    version: int
    embed_model_name: str
    items: List[str]
    freqs: List[float]
    n_neighbors: int
    enable_lof: bool
    lof_contamination: float
    embeddings: Optional[Any] = None


def save_index(index: IngredientIndex, path: str) -> None:
    emb: Optional[Any] = None
    if index.embeddings is not None and isinstance(index.embeddings, np.ndarray):
        emb = index.embeddings.astype(np.float32)

    obj = SavedModel(
        version=3,
        embed_model_name=index.embed_model_name,
        items=index.items,
        freqs=[float(x) for x in index.freqs.tolist()],
        n_neighbors=index.n_neighbors,
        enable_lof=index.enable_lof,
        lof_contamination=index.lof_contamination,
        embeddings=emb,
    )

    with open(path, "wb") as f:
        pickle.dump(asdict(obj), f)

    LOG.info("Saved index to %s", path)


def _validate_saved_model(obj: SavedModel) -> None:
    if int(obj.version) != 3:
        raise ValueError(f"Invalid model file: unsupported version {obj.version} (expected 3).")
    if not isinstance(obj.items, list) or not obj.items:
        raise ValueError("Invalid model file: items missing.")
    if not isinstance(obj.freqs, list) or len(obj.freqs) != len(obj.items):
        raise ValueError("Invalid model file: freqs length mismatch.")
    if len(set(obj.items)) != len(obj.items):
        raise ValueError("Invalid model file: items must be unique.")
    for x in obj.freqs:
        fx = float(x)
        if fx < 0.0 or fx > 1.0 or math.isnan(fx):
            raise ValueError("Invalid model file: freqs out of range.")
    if int(obj.n_neighbors) < 2:
        raise ValueError("Invalid model file: n_neighbors invalid.")


def load_index(path: str) -> IngredientIndex:
    LOG.warning("Loading a pickle index file: %s", path)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Invalid model file: expected a dict payload.")

    obj = SavedModel(**obj)
    _validate_saved_model(obj)

    idx = IngredientIndex(
        items=list(obj.items),
        freqs=np.array(obj.freqs, dtype=np.float32),
        embed_model_name=obj.embed_model_name,
        n_neighbors=int(obj.n_neighbors),
        enable_lof=bool(obj.enable_lof),
        lof_contamination=float(obj.lof_contamination),
    )

    precomputed: Optional[np.ndarray] = None
    if obj.embeddings is not None:
        try:
            precomputed = np.asarray(obj.embeddings, dtype=np.float32)
        except Exception:
            precomputed = None

    idx.fit(precomputed_embeddings=precomputed, allow_missing_embedder=True)
    LOG.info("Loaded index from %s", path)
    return idx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="appropriateness.py",
        description=(
            "Evaluate whether items are appropriate in a context.\n\n"
            "Examples:\n"
            "  python appropriateness.py --context \"pasta ingredients\" cheetos tofu\n"
            "  python appropriateness.py --dataset recipes.csv --column ingredients --context \"pasta ingredients\" --items-text \"cheetos, tofu\"\n"
            "  python appropriateness.py --model saved.idx --context \"pasta ingredients\" cheetos\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("context_pos", nargs="?", help="Context text (quoted if multi-word). Prefer --context for clarity.")
    p.add_argument(
        "items",
        nargs="*",
        help="Items to evaluate (repeatable). For multiple items in one argument, use separators (comma/semicolon/pipe/newline or ' / ') or --items-text.",
    )
    p.add_argument("--context", default=None, help="Context text (preferred; overrides positional context).")

    p.add_argument("-i", "--items-text", default=None, help="Items in one string (separators supported).")

    p.add_argument("-d", "--dataset", default=None, help="CSV path (example rows).")
    p.add_argument("-c", "--column", default="items", help="Dataset column containing item lists (default: items).")

    p.add_argument("-m", "--model", default=None, help="Load a previously saved index file.")
    p.add_argument("--save", default=None, help="Save the built index to this file.")

    p.add_argument("--embed", default="all-MiniLM-L6-v2", help="SentenceTransformer model (default: all-MiniLM-L6-v2).")
    p.add_argument("-k", "--k", type=int, default=5, help="Neighbors to show/use (default: 5).")

    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="normal", help="Decision preset (default: normal).")
    p.add_argument("--min-sim", type=float, default=None, help="Override minimum similarity (0..1).")
    p.add_argument("--min-freq", type=float, default=None, help="Override minimum frequency (0..1 of used rows).")
    p.add_argument("--allow-outlier", action="store_true", help="Allow LOF outliers (more permissive).")
    p.add_argument("--no-lof", action="store_true", help="Disable outlier detection.")

    p.add_argument("--llm", action="store_true", help="Enable LLM veto (requires OPENAI_API_KEY and openai SDK).")
    p.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model (default: gpt-4o-mini).")
    p.add_argument("--llm-min-p", type=float, default=None, help="LLM minimum p to avoid veto (override preset).")

    p.add_argument("--explain", action="store_true", help="Show neighbors and signal breakdown.")
    p.add_argument("--json", action="store_true", help="Print JSON output (one per line).")
    p.add_argument("-q", "--quiet", action="store_true", help="Reduce logs.")
    return p


def _warn_if_context_looks_like_item(context: str, items: List[str], used_positional: bool) -> None:
    if not used_positional:
        return
    if not context or " " in context:
        return
    if len(items) == 0:
        return
    LOG.warning(
        "Context was provided positionally as a single token (%r). If you intended a multi-word context, use --context.",
        context,
    )


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(level=logging.WARNING if args.quiet else logging.INFO)
    if args.quiet:
        LOG.setLevel(logging.WARNING)

    used_positional_context = False
    context = ""
    items: List[str] = []

    if args.context is not None:
        context = str(args.context).strip()
        if args.context_pos:
            items.extend(split_items(args.context_pos))
    else:
        context = str(args.context_pos or "").strip()
        used_positional_context = bool(context)

    if args.items_text:
        items.extend(split_items(args.items_text))

    if args.items:
        for arg in args.items:
            items.extend(split_items(arg))

    items = [x.strip() for x in items if x and x.strip()]

    _warn_if_context_looks_like_item(context, items, used_positional_context)

    if not context:
        context = input("Context: ").strip()

    if not items:
        raw = input("Items (separators supported): ").strip()
        items = split_items(raw)

    if not items:
        LOG.error("No items provided.")
        sys.exit(1)

    if args.llm and not os.getenv("OPENAI_API_KEY"):
        LOG.warning("OPENAI_API_KEY is not set; LLM veto will not run.")

    try:
        if args.model:
            idx = load_index(args.model)
            stats: Optional[IndexStats] = None
        else:
            if args.dataset:
                path = os.path.expanduser(args.dataset)
                if not os.path.exists(path):
                    LOG.error("Dataset not found: %s", path)
                    sys.exit(1)
                df = pd.read_csv(path)
            else:
                df = pd.DataFrame({
                    args.column: [
                        "spaghetti, garlic, tomato sauce, basil, olive oil, parmesan",
                        "penne, garlic, tomato, onion, basil, olive oil, feta cheese",
                        "fettuccine, cream, garlic, parmesan, black pepper, parsley",
                        "linguine, garlic, shrimp, lemon, butter, parsley",
                        "rigatoni, sausage, tomato sauce, onion, bell pepper, parmesan",
                    ]
                })

            idx, stats = IngredientIndex.from_dataset(
                df=df,
                column=args.column,
                embed_model_name=args.embed,
                n_neighbors=max(args.k, 10),
                enable_lof=(not args.no_lof),
            )

            if args.save:
                save_index(idx, args.save)

        base = PRESETS[args.preset]
        min_sim = base.min_sim
        min_freq = base.min_freq

        if not args.model and stats is not None:
            min_sim = clamp(stats.suggested_min_sim, base.min_sim - 0.05, base.min_sim + 0.05)
            min_freq = clamp(stats.suggested_min_freq, base.min_freq - 0.02, base.min_freq + 0.06)

        if args.min_sim is not None:
            min_sim = clamp(float(args.min_sim), 0.0, 1.0)
        if args.min_freq is not None:
            min_freq = clamp(float(args.min_freq), 0.0, 1.0)

        llm_enabled = bool(args.llm)
        llm_min_p = base.llm_min_p if args.llm_min_p is None else clamp(float(args.llm_min_p), 0.0, 1.0)

        thresholds = Thresholds(
            min_sim=min_sim,
            min_freq=min_freq,
            allow_outlier=bool(args.allow_outlier) or base.allow_outlier,
            llm_enabled=llm_enabled,
            llm_min_p=llm_min_p,
        )

        evaluator = AppropriatenessEvaluator(
            context=context,
            index=idx,
            thresholds=thresholds,
            k_neighbors=args.k,
            llm_model=args.llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        for it in items:
            ev = evaluator.evaluate(it, explain=args.explain)
            if args.json:
                print(json.dumps(asdict(ev), ensure_ascii=False))
            else:
                line = f"{ev.item}: {ev.decision} (score={ev.score:.3f})"
                if ev.reason:
                    line += f" — {ev.reason}"
                print(line)
                if args.explain and ev.neighbors:
                    print("  Neighbors:")
                    for n, sim, fr in ev.neighbors:
                        print(f"    - {n:<28}  sim={sim:.3f}  freq={fr:.3f}")

    except Exception as e:
        LOG.error("Fatal error: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
