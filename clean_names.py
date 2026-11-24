"""
clean_names: normalize arbitrary strings into consistent, identifier-friendly names.

Pipeline (one pass, each stage a small pure function):

    NFC -> replacements -> transliteration -> split into segments (sep_in)
        -> per segment: numerals -> tokenize into words -> render in `case`
        -> drop empty segments -> de-duplicate segments -> join with sep_out

Design notes
  * Words are found by a single tokenizer (non-alphanumerics and camelCase
    boundaries), so every case style sees the same words and the output no
    longer depends on whether optional packages happen to be installed.
  * Greek, Cyrillic and common Latin letters that Unicode does not decompose
    (ß, æ, ø, ł, ...) are transliterated by built-in tables. `unidecode`, if
    installed, is only used to rescue letters that would otherwise be dropped.
  * Options are resolved and validated once into a `_Plan`, so
    `clean_names_table` pays that cost once, not once per row.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # Optional: rescues letters from scripts without a built-in table (CJK, Arabic, ...).
    from unidecode import unidecode  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    unidecode = None

__all__ = ["clean_names", "clean_names_table", "apply_replacements", "remove_duplicate_names"]

# A library must not configure logging; the application decides where records go.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SepIn = Union[str, Sequence[str], None]
Words = List[str]


# ----------------------------- Option vocabularies -----------------------------

_CASE_ALIASES: Dict[str, str] = {
    **dict.fromkeys(("snake", "sc", "s"), "snake"),
    **dict.fromkeys(("camel", "cc", "lowercamel"), "camel"),
    **dict.fromkeys(("pascal", "pc", "uppercamel"), "pascal"),
    **dict.fromkeys(("kebab", "kc"), "kebab"),
    **dict.fromkeys(("space", "sp"), "space"),
    **dict.fromkeys(("title", "t"), "title"),
    **dict.fromkeys(("upper", "u", "screamingsnake"), "upper"),
    **dict.fromkeys(("lower", "l"), "lower"),
    **dict.fromkeys(("sentence", "sent"), "sentence"),
    "oldjanitor": "old_janitor",
}

_NUMERAL_ALIASES: Dict[str, str] = {
    **dict.fromkeys(("asis", "keep", "k"), "asis"),
    **dict.fromkeys(("remove", "rm", "strip"), "remove"),
    **dict.fromkeys(("spell", "words", "w"), "spell"),
}

_STEP_ALIASES: Dict[str, str] = {
    **dict.fromkeys(("latin", "nfkd", "normalize"), "latin"),
    "ascii": "ascii",
    **dict.fromkeys(("greek", "el"), "greek"),
    **dict.fromkeys(("cyrillic", "ru", "uk", "bg"), "cyrillic"),
}

_DEFAULT_REPLACE: Mapping[str, str] = {"'": "", '"': "", "%": "_percent_", "#": "_number_", "µ": "u"}
_OLD_JANITOR_REPLACE: Mapping[str, str] = {"'": "", '"': "", "%": "percent", "#": "number", "µ": "u"}
_MICRO_SIGN = "µ"  # µ: "micro", conventionally "u" (ug, uL). NFKD would turn it into Greek mu -> "m".


# ----------------------------- Public API -----------------------------

def clean_names(
    string: str,
    case: str = "snake",
    replace: Optional[Mapping[str, str]] = None,
    ascii: bool = True,
    use_make_names: bool = True,
    allow_dupes: bool = False,
    sep_in: SepIn = r"\.",
    transliterations: Union[str, Sequence[str]] = "Latin-ASCII",
    parsing_option: Union[int, str] = 1,
    numerals: str = "asis",
    sep_out: str = ".",
    replace_regex: bool = False,
    transliterate: Optional[bool] = None,
    warn_unhandled_unicode: bool = True,
) -> str:
    """
    Clean one string into a name, e.g. ``"Sales ($) 2024"`` -> ``"sales_2024"``.

    case            snake | camel | pascal | kebab | space | title | upper | lower |
                    sentence | old_janitor. Aliases such as "snake_case", "sc",
                    "kebab-case" are accepted.
    replace         Literal substring -> replacement (longest key wins). Defaults map
                    quotes away, % -> percent, # -> number, µ -> u.
    replace_regex   Treat `replace` keys as regexes, applied in order.
    ascii           Guarantee ASCII output (transliterating where possible).
    transliterate   None: transliterate iff ascii. True/False: force on/off.
    transliterations Steps in order: "latin", "greek", "cyrillic", "ascii". Accepts
                    "Latin-ASCII", "greek, ascii", ["Greek", "ASCII"], ...
    sep_in          Segment separators kept in the output (default "."). A string is a
                    set of single characters (r"\\." style regex fragments still work);
                    a list may hold multi-character tokens. "" or [] disables.
    sep_out         Joins segments in the output (default ".").
    parsing_option  1 splits on sep_in; any other int treats the string as one segment.
    numerals        asis | remove | spell (spelling needs `num2words`).
    use_make_names  Spell a name that is only digits ("123" -> "one_hundred_...").
    allow_dupes     False: repeated segments get suffixes ("a.a" -> "a.a_2").

    Raises ValueError for list-like input or unknown option values.
    """
    plan = _build_plan(
        case=case, replace=replace, ascii=ascii, use_make_names=use_make_names,
        allow_dupes=allow_dupes, sep_in=sep_in, transliterations=transliterations,
        parsing_option=parsing_option, numerals=numerals, sep_out=sep_out,
        replace_regex=replace_regex, transliterate=transliterate,
        warn_unhandled_unicode=warn_unhandled_unicode,
    )
    return plan(_coerce(string, strict=True))


def clean_names_table(
    values: Iterable[Union[str, int, float, None]],
    *,
    case: str = "snake",
    replace: Optional[Mapping[str, str]] = None,
    ascii: bool = True,
    use_make_names: bool = True,
    allow_dupes: bool = False,
    sep_in: SepIn = r"\.",
    transliterations: Union[str, Sequence[str]] = "Latin-ASCII",
    parsing_option: Union[int, str] = 1,
    numerals: str = "asis",
    sep_out: str = ".",
    replace_regex: bool = False,
    transliterate: Optional[bool] = None,
    warn_unhandled_unicode: bool = True,
    skip_none: bool = False,
    unique: bool = False,
) -> List[Tuple[str, str]]:
    """
    Return ``[(original, cleaned), ...]`` in input order.

    Non-strings are stringified (123 -> "123"); None becomes "" unless skip_none=True.
    unique=True also makes the cleaned column unique across rows ("a b", "A-B" ->
    "a_b", "a_b_2"), which is what you want when the values are column headers.
    Options are validated once, before any row is processed.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("`values` must be an iterable of values, not a single string.")
    plan = _build_plan(
        case=case, replace=replace, ascii=ascii, use_make_names=use_make_names,
        allow_dupes=allow_dupes, sep_in=sep_in, transliterations=transliterations,
        parsing_option=parsing_option, numerals=numerals, sep_out=sep_out,
        replace_regex=replace_regex, transliterate=transliterate,
        warn_unhandled_unicode=warn_unhandled_unicode,
    )
    originals = ["" if v is None else _coerce(v) for v in values if not (v is None and skip_none)]
    cleaned = [plan(o) for o in originals]
    if unique:
        cleaned = remove_duplicate_names(cleaned)
    return list(zip(originals, cleaned))


def apply_replacements(string: str, replace_dict: Mapping[str, str], replace_regex: bool = False) -> str:
    """
    Literal mode (default): one pass, longest key first, empty keys ignored.
    Regex mode: each (pattern, repl) applied in order; backreferences work.
    """
    return _replacer(tuple(replace_dict.items()), replace_regex)(string)


def remove_duplicate_names(names: Sequence[str]) -> List[str]:
    """
    Suffix repeats with _2, _3, ... without colliding with any other name.

    ["a", "a", "a"]   -> ["a", "a_2", "a_3"]
    ["a", "a", "a_2"] -> ["a", "a_3", "a_2"]   (a name that was already unique is never renamed)
    """
    taken = set(names)
    seen: set = set()
    next_suffix: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
            continue
        k = next_suffix.get(name, 2)
        while f"{name}_{k}" in taken:
            k += 1
        candidate = f"{name}_{k}"
        next_suffix[name] = k + 1
        taken.add(candidate)
        out.append(candidate)
    return out


# ----------------------------- Plan -----------------------------

@dataclass(frozen=True)
class _Plan:
    replace: Callable[[str], str]
    transliterate: Optional[Callable[[str], str]]
    split: Optional[Callable[[str], List[str]]]
    numerals: Optional[Callable[[str], str]]
    spell_all_digits: bool
    split_camel: bool
    render: Callable[[Words], str]
    dedupe: bool
    sep_out: str
    warn_micro: bool

    def __call__(self, text: str) -> str:
        original = text
        text = unicodedata.normalize("NFC", text)
        if self.warn_micro and _MICRO_SIGN in text:
            logger.warning("The micro sign (µ) is present and has no replacement rule; it will not become 'u'.")
        text = self.replace(text)
        if self.transliterate:
            text = self.transliterate(text)

        segments = [s for s in (self.split(text) if self.split else [text]) if s and not s.isspace()]
        if self.spell_all_digits and len(segments) == 1 and segments[0].strip().isdecimal():
            segments = [_spell(segments[0].strip())]
        if self.numerals:
            segments = [self.numerals(s) for s in segments]

        names = [self.render(words) for s in segments if (words := _words(s, self.split_camel))]
        if self.dedupe:
            names = remove_duplicate_names(names)
        result = self.sep_out.join(names)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("clean_names: %r -> %r", original, result)
        return result


def _build_plan(
    *, case: str, replace: Optional[Mapping[str, str]], ascii: bool, use_make_names: bool,
    allow_dupes: bool, sep_in: SepIn, transliterations: Union[str, Sequence[str]],
    parsing_option: Union[int, str], numerals: str, sep_out: str, replace_regex: bool,
    transliterate: Optional[bool], warn_unhandled_unicode: bool,
) -> _Plan:
    case = _normalize_case(case)
    numerals = _lookup(numerals, _NUMERAL_ALIASES, "numerals")
    if not isinstance(sep_out, str):
        raise TypeError(f"`sep_out` must be a str, got {type(sep_out).__name__}.")
    old_janitor = case == "old_janitor"

    if replace is None:
        replace = _OLD_JANITOR_REPLACE if old_janitor else _DEFAULT_REPLACE

    steps: Tuple[str, ...] = ()
    if ascii if transliterate is None else transliterate:
        steps = _parse_steps(_freeze(transliterations))
        if ascii and steps[-1:] != ("ascii",):
            steps += ("ascii",)  # ascii=True is a guarantee, not a hint

    return _Plan(
        replace=_replacer(tuple(replace.items()), replace_regex),
        transliterate=_transliterator(steps, warn_unhandled_unicode) if steps else None,
        split=None if old_janitor or _normalize_parsing_option(parsing_option) != 1 else _splitter(_freeze(sep_in)),
        numerals={"asis": None, "remove": _remove_digits, "spell": _spell_digits}[numerals],
        spell_all_digits=use_make_names and numerals == "asis",
        split_camel=not old_janitor,
        render=_RENDERERS["snake" if old_janitor else case],
        dedupe=not allow_dupes and not old_janitor,
        sep_out=sep_out,
        warn_micro=not any(_MICRO_SIGN in key for key in replace),
    )


# ----------------------------- Replacement & separators -----------------------------

def _identity(s: str) -> str:
    return s


@lru_cache(maxsize=256)
def _replacer(items: Tuple[Tuple[str, str], ...], regex: bool) -> Callable[[str], str]:
    if regex:
        compiled = [(re.compile(pattern), repl) for pattern, repl in items]

        def apply_all(s: str) -> str:
            for pattern, repl in compiled:
                s = pattern.sub(repl, s)
            return s

        return apply_all

    table = {k: v for k, v in items if k}  # an empty key would match between every character
    if not table:
        return _identity
    # Longest first, so {"%": .., "%%": ..} matches "%%" as a unit regardless of dict order.
    pattern = re.compile("|".join(map(re.escape, sorted(table, key=len, reverse=True))))
    return lambda s: pattern.sub(lambda m: table[m[0]], s)


@lru_cache(maxsize=64)
def _splitter(sep_in: Union[str, Tuple[str, ...], None]) -> Optional[Callable[[str], List[str]]]:
    if sep_in is None:
        sep_in = "."
    if isinstance(sep_in, str):
        if not sep_in:
            return None
        pattern = None
        if "\\" in sep_in:  # legacy: a character-class fragment such as r"\." or r"\.\-"
            try:
                pattern = re.compile(f"[{sep_in}]+")
            except re.error:
                pass  # e.g. a lone backslash: fall back to treating it literally
        pattern = pattern or re.compile(f"[{re.escape(sep_in)}]+")
    else:
        tokens = sorted({s for s in sep_in if s}, key=len, reverse=True)
        if not tokens:
            return None
        pattern = re.compile(f"(?:{'|'.join(map(re.escape, tokens))})+")
    return pattern.split


def _freeze(value):
    """Lists are unhashable; make option values usable as lru_cache keys."""
    return tuple(str(v) for v in value) if isinstance(value, (list, tuple, set, frozenset)) else value


# ----------------------------- Numerals -----------------------------

_DIGITS = re.compile(r"\d+")


def _remove_digits(s: str) -> str:
    return _DIGITS.sub("", s)


def _spell_digits(s: str) -> str:
    return _DIGITS.sub(lambda m: _spell(m[0]), s)


def _spell(digits: str) -> str:
    # Padded with spaces so "Chapter10" spells to "chapter_ten", not "chapterten".
    return f" {_num2words()(int(digits))} "


@lru_cache(maxsize=None)
def _num2words():
    try:
        from num2words import num2words  # type: ignore
    except ImportError as e:
        raise ModuleNotFoundError(
            "num2words is required to spell numbers (numerals='spell', or an all-digit name "
            "with use_make_names=True). Install it with `pip install num2words`."
        ) from e
    return num2words


# ----------------------------- Tokenizing & rendering -----------------------------

# Same boundaries as inflection.underscore: "fooBar" | "foo2Bar" | "HTTPServer".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ASCII_WORDS = re.compile(r"[A-Za-z0-9]+").findall


def _words(text: str, split_camel: bool) -> Words:
    if split_camel:
        text = _CAMEL_BOUNDARY.sub(" ", text)
    if text.isascii():
        return _ASCII_WORDS(text)
    # Unicode path: keep combining marks inside words (Devanagari, Thai, NFD input, ...).
    return "".join(
        c if c.isalnum() or unicodedata.category(c)[0] == "M" else " " for c in text
    ).split()


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:].lower()  # unlike str.title(), "2b" stays "2b"


def _lower_join(sep: str) -> Callable[[Words], str]:
    return lambda ws: sep.join(w.lower() for w in ws)


_RENDERERS: Dict[str, Callable[[Words], str]] = {
    "snake": _lower_join("_"),
    "lower": _lower_join("_"),
    "kebab": _lower_join("-"),
    "space": _lower_join(" "),
    "upper": lambda ws: "_".join(w.upper() for w in ws),
    "title": lambda ws: "_".join(map(_cap, ws)),
    "sentence": lambda ws: "_".join([_cap(ws[0]), *(w.lower() for w in ws[1:])]),
    "camel": lambda ws: ws[0].lower() + "".join(map(_cap, ws[1:])),
    "pascal": lambda ws: "".join(map(_cap, ws)),
}


# ----------------------------- Transliteration -----------------------------

_GREEK = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "h", "θ": "th",
    "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o",
}
_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "є": "ye", "і": "i", "ї": "yi", "ґ": "g", "ў": "u",
}
# Latin letters that NFKD does not decompose, so a plain ASCII filter would delete them.
_LATIN_EXTRA = {
    "ß": "ss", "ẞ": "Ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ð": "d",
    "ł": "l", "þ": "th", "ı": "i", "ħ": "h", "ŧ": "t", "ŋ": "ng",
}
_GREEK_RUN = re.compile(r"[Ͱ-Ͽἀ-῿]+")  # Greek + polytonic Greek
_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")


def _table_transliterator(lower: Mapping[str, str]) -> Callable[[str], str]:
    """
    Character-table transliterator that derives uppercase entries and picks the
    case of multi-letter outputs from context, so camelCase splitting still works:
    "Θέα" -> "Thea", "ΘΕΑ" -> "THEA", "ΑΘ" -> "ATH".
    """
    table = dict(lower)
    for k, v in lower.items():
        upper = k.upper()
        if len(upper) == 1 and upper != k and not upper.isascii():
            table.setdefault(upper, v.capitalize())
    pattern = re.compile(f"[{''.join(map(re.escape, table))}]")

    def repl(m: re.Match) -> str:
        out = table[m[0]]
        if len(out) > 1 and out[0].isupper():
            s, i, j = m.string, m.start(), m.end()
            nxt, prv = s[j:j + 1], s[i - 1:i] if i else ""
            if nxt.isupper() or (prv.isupper() and not nxt.islower()):
                return out.upper()
        return out

    return lambda s: s if s.isascii() else pattern.sub(repl, s)


_greek_table = _table_transliterator(_GREEK)
_cyrillic = _table_transliterator(_CYRILLIC)
_latin_extra = _table_transliterator(_LATIN_EXTRA)


def _strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _latin(s: str) -> str:
    """NFKD compatibility folding ("ﬁ" -> "fi", "²" -> "2") and accent removal on Latin letters only."""
    if s.isascii():
        return s
    out: List[str] = []
    base_is_ascii = False
    for c in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(c):
            if base_is_ascii:
                continue  # drop accents on Latin letters; keep them on й, ά, हि, ...
        else:
            base_is_ascii = c.isascii()
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def _greek(s: str) -> str:
    return _GREEK_RUN.sub(lambda m: _greek_table(_strip_marks(m[0])), s)


def _ascii(s: str, warn: bool) -> str:
    s = _latin(_latin_extra(_greek(_cyrillic(s))))  # Cyrillic first: й/ё must not lose their marks
    if s.isascii():
        return s
    if unidecode is not None:
        # Rescue only runs containing letters/digits; symbol-only runs become word breaks
        # either way, so unidecode changes output only where text would otherwise be lost.
        return _NON_ASCII_RUN.sub(lambda m: unidecode(m[0]) if any(map(str.isalnum, m[0])) else " ", s)
    if warn:
        lost = sorted({c for c in s if not c.isascii() and c.isalnum()})
        if lost:
            logger.warning("Dropped characters with no ASCII transliteration: %r "
                           "(install `unidecode` or add `replace` rules).", "".join(lost))
    return _NON_ASCII_RUN.sub(" ", s)  # a space, not "", so dropped text still separates words


_STEPS: Dict[str, Callable[[str], str]] = {"latin": _latin, "greek": _greek, "cyrillic": _cyrillic}


@lru_cache(maxsize=64)
def _transliterator(steps: Tuple[str, ...], warn: bool) -> Callable[[str], str]:
    funcs = [(lambda s: _ascii(s, warn)) if step == "ascii" else _STEPS[step] for step in steps]

    def run(s: str) -> str:
        for f in funcs:
            s = f(s)
        if warn and "ascii" not in steps and not s.isascii():
            logger.warning("Characters left untransliterated: %r",
                           "".join(sorted({c for c in s if not c.isascii()})))
        return s

    return run


@lru_cache(maxsize=64)
def _parse_steps(spec: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    parts = spec if isinstance(spec, tuple) else re.split(r"[\s,|+;-]+", str(spec))
    return tuple(_lookup(p, _STEP_ALIASES, "transliteration step") for p in parts if p.strip())


# ----------------------------- Input normalization -----------------------------

def _coerce(value, strict: bool = False) -> str:
    if strict and isinstance(value, (list, tuple, set, frozenset, dict)):
        raise ValueError("`string` must be a single string; use clean_names_table() for many values.")
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _lookup(value: str, aliases: Mapping[str, str], what: str) -> str:
    key = str(value).strip().lower()
    if key in aliases:
        return aliases[key]
    valid = ", ".join(sorted(set(aliases.values())))
    raise ValueError(f"Unknown {what} {value!r}; expected one of: {valid}.")


def _normalize_case(case: str) -> str:
    key = re.sub(r"[\s_-]+", "", str(case).lower())
    if key.endswith("case") and key != "case":
        key = key[:-4]  # "snake_case", "kebab-case", "UPPERCASE"
    return _lookup(key, _CASE_ALIASES, "case")


def _normalize_parsing_option(option: Union[int, str]) -> int:
    if isinstance(option, int):
        return option
    s = str(option).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    raise ValueError(f"`parsing_option` must be an integer, got {option!r}.")
