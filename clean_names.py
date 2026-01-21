import re
import unicodedata
import logging
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, Iterable

# Optional dependencies
try:
    from unidecode import unidecode  # type: ignore
except ImportError:
    unidecode = None

try:
    from inflection import underscore as _inflection_underscore  # type: ignore
except ImportError:
    _inflection_underscore = None


# ----------------------------- Logging -----------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


# ----------------------------- Transliteration -----------------------------

GREEK_TRANSLIT_RULES: Dict[str, str] = {
    "α": "a", "β": "b", "γ": "g", "δ": "d",
    "ε": "e", "ζ": "z", "η": "h", "θ": "th",
    "ι": "i", "κ": "k", "λ": "l", "μ": "m",
    "ν": "n", "ξ": "x", "ο": "o", "π": "p",
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t",
    "υ": "u", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o", "ά": "a", "έ": "e", "ί": "i",
    "ό": "o", "ύ": "u", "ή": "h", "ώ": "o",
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D",
    "Ε": "E", "Ζ": "Z", "Η": "H", "Θ": "TH",
    "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M",
    "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P",
    "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y",
    "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O",
    "Ά": "A", "Έ": "E", "Ί": "I", "Ό": "O",
    "Ύ": "Y", "Ή": "H", "Ώ": "O",
}
GREEK_TRANSLIT_TABLE = str.maketrans(GREEK_TRANSLIT_RULES)


# ----------------------------- Friendly aliases -----------------------------

_CASE_ALIASES: Dict[str, str] = {
    "snake": "snake",
    "snake_case": "snake",
    "snakecase": "snake",
    "sc": "snake",
    "s": "snake",

    "camel": "camel",
    "camelcase": "camel",
    "cc": "camel",

    "pascal": "pascal",
    "pascalcase": "pascal",
    "pc": "pascal",

    "kebab": "kebab",
    "kebab-case": "kebab",
    "kc": "kebab",

    "space": "space",
    "sp": "space",

    "title": "title",
    "t": "title",

    "upper": "upper",
    "u": "upper",

    "lower": "lower",
    "l": "lower",

    "sentence": "sentence",
    "sent": "sentence",

    "old_janitor": "old_janitor",
}

_NUMERAL_ALIASES: Dict[str, str] = {
    "asis": "asis",
    "as-is": "asis",
    "keep": "asis",
    "k": "asis",

    "remove": "remove",
    "rm": "remove",
    "strip": "remove",

    "spell": "spell",
    "words": "spell",
    "w": "spell",
}


# ----------------------------- Public API -----------------------------

def clean_names(
    string: str,
    case: str = "snake",
    replace: Optional[Dict[str, str]] = None,
    ascii: bool = True,
    use_make_names: bool = True,
    allow_dupes: bool = False,
    sep_in: Union[str, Sequence[str], None] = r"\.",
    transliterations: Union[str, Sequence[str]] = "Latin-ASCII",
    parsing_option: Union[int, str] = 1,
    numerals: str = "asis",
    # Enhancements (optional; defaults preserve existing behavior)
    sep_out: str = ".",
    replace_regex: bool = False,
    transliterate: Optional[bool] = None,
    warn_unhandled_unicode: bool = True,
) -> str:
    """
    Clean and standardize a given string into a valid name.

    Backward-compatible defaults:
      - Returns a single cleaned string.
      - Raises ValueError if `string` is a list (preserved).

    Enhancements:
      - sep_in can be "." (literal), r"\\." (like before), a list like [".", "/"], or None.
      - transliterations accepts list/tuple or easy strings like "Greek, ASCII" or "latin ascii".
      - case/numerals accept friendly aliases (e.g., case="snake_case", numerals="keep").
      - sep_out controls the segment separator produced by parsing_option=1 (default ".").
      - replace_regex=True enables regex replacement keys (default False).
      - transliterate controls whether transliteration runs even when ascii=False:
            None => old behavior (transliterate iff ascii=True)
            True => always transliterate using 'transliterations'
            False => never transliterate
    """
    if isinstance(string, list):
        raise ValueError("`string` must not be a list, it should be a single string.")
    if not isinstance(string, str):
        string = str(string)

    case_norm = _normalize_case(case)
    numerals_norm = _normalize_numerals(numerals)
    parse_opt = _normalize_parsing_option(parsing_option)
    transliterate_flag = ascii if transliterate is None else bool(transliterate)

    logger.debug(
        "Initial string: %s | case=%s numerals=%s ascii=%s use_make_names=%s allow_dupes=%s sep_in=%r sep_out=%r "
        "transliterations=%r parsing_option=%s replace_regex=%s transliterate=%s",
        string, case_norm, numerals_norm, ascii, use_make_names, allow_dupes, sep_in, sep_out,
        transliterations, parse_opt, replace_regex, transliterate_flag
    )

    return _clean_single(
        string=string,
        case=case_norm,
        replace=replace,
        ascii=ascii,
        use_make_names=use_make_names,
        allow_dupes=allow_dupes,
        sep_in=sep_in,
        transliterations=transliterations,
        parsing_option=parse_opt,
        numerals=numerals_norm,
        sep_out=sep_out,
        replace_regex=replace_regex,
        transliterate_flag=transliterate_flag,
        warn_unhandled_unicode=warn_unhandled_unicode,
    )


def clean_names_table(
    values: Iterable[Union[str, int, float, None]],
    *,
    case: str = "snake",
    replace: Optional[Dict[str, str]] = None,
    ascii: bool = True,
    use_make_names: bool = True,
    allow_dupes: bool = False,
    sep_in: Union[str, Sequence[str], None] = r"\.",
    transliterations: Union[str, Sequence[str]] = "Latin-ASCII",
    parsing_option: Union[int, str] = 1,
    numerals: str = "asis",
    sep_out: str = ".",
    replace_regex: bool = False,
    transliterate: Optional[bool] = None,
    warn_unhandled_unicode: bool = True,
    skip_none: bool = False,
) -> List[Tuple[str, str]]:
    """
    Return a 2-column "data table" mapping original -> renamed.

    Output format: List[Tuple[original_value, renamed_value]]

    Notes:
      - Order is preserved (stable, cell-to-cell friendly).
      - Non-strings are stringified (e.g., 123 -> "123") for practical spreadsheet usage.
      - None values can be skipped via skip_none=True.
    """
    case_norm = _normalize_case(case)
    numerals_norm = _normalize_numerals(numerals)
    parse_opt = _normalize_parsing_option(parsing_option)
    transliterate_flag = ascii if transliterate is None else bool(transliterate)

    table: List[Tuple[str, str]] = []
    for v in values:
        if v is None and skip_none:
            continue
        original = "" if v is None else (v if isinstance(v, str) else str(v))
        renamed = _clean_single(
            string=original,
            case=case_norm,
            replace=replace,
            ascii=ascii,
            use_make_names=use_make_names,
            allow_dupes=allow_dupes,
            sep_in=sep_in,
            transliterations=transliterations,
            parsing_option=parse_opt,
            numerals=numerals_norm,
            sep_out=sep_out,
            replace_regex=replace_regex,
            transliterate_flag=transliterate_flag,
            warn_unhandled_unicode=warn_unhandled_unicode,
        )
        table.append((original, renamed))
    return table


# ----------------------------- Internal pipeline -----------------------------

def _clean_single(
    *,
    string: str,
    case: str,
    replace: Optional[Dict[str, str]],
    ascii: bool,
    use_make_names: bool,
    allow_dupes: bool,
    sep_in: Union[str, Sequence[str], None],
    transliterations: Union[str, Sequence[str]],
    parsing_option: int,
    numerals: str,
    sep_out: str,
    replace_regex: bool,
    transliterate_flag: bool,
    warn_unhandled_unicode: bool,
) -> str:
    # Special branch preserved (output behavior unchanged).
    if case == "old_janitor":
        replace_dict = {"'": "", '"': "", "%": "percent", "#": "number"}
        warn_about_micro_symbol(string, replace_dict)
        cleaned = apply_replacements(string, replace_dict, replace_regex=False)
        cleaned = make_valid_names(cleaned, numerals)
        cleaned = re.sub(r"[.]+", "_", cleaned)
        cleaned = re.sub(r"[_]+", "_", cleaned)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"_$", "", cleaned)
        if not allow_dupes:
            parts = cleaned.split("_")
            cleaned = "_".join(remove_duplicate_names(parts))
        logger.debug("Final string (old_janitor): %s", cleaned)
        return cleaned

    replace_dict = replace if replace is not None else {"'": "", '"': "", "%": "_percent_", "#": "_number_"}
    warn_about_micro_symbol(string, replace_dict)

    cleaned = apply_replacements(string, replace_dict, replace_regex=replace_regex)
    logger.debug("After replacements: %s", cleaned)

    cleaned = apply_transliterations(
        cleaned,
        transliterations=transliterations,
        do_transliterate=transliterate_flag,
        warn_unhandled_unicode=warn_unhandled_unicode,
    )
    logger.debug("After transliterations: %s", cleaned)

    cleaned = sanitize_string(cleaned, sep_in=sep_in, sep_out=sep_out, parsing_option=parsing_option)
    logger.debug("After sanitization: %s", cleaned)

    if numerals != "asis":
        cleaned = handle_numerals(cleaned, numerals)
        logger.debug("After handling numerals: %s", cleaned)

    if use_make_names:
        cleaned = make_valid_names(cleaned, numerals)
        logger.debug("After making valid names: %s", cleaned)

    cleaned = apply_case_conversion(cleaned, case=case, segment_sep=sep_out)
    logger.debug("After case conversion: %s", cleaned)

    if not allow_dupes:
        parts = cleaned.split(sep_out) if sep_out else cleaned.split(".")
        cleaned = (sep_out if sep_out else ".").join(remove_duplicate_names(parts))
        logger.debug("After removing duplicates: %s", cleaned)

    return cleaned


# ----------------------------- Core helpers -----------------------------

def apply_replacements(string: str, replace_dict: Mapping[str, str], replace_regex: bool = False) -> str:
    """
    Apply replacement rules.

    - replace_regex=False (default): keys are treated literally and replaced in a single pass.
    - replace_regex=True: keys are treated as regex patterns and applied sequentially (in dict order).
    """
    if not replace_dict:
        return string

    if replace_regex:
        # Sequential (predictable): supports capture groups, backrefs, etc.
        for pattern, repl in replace_dict.items():
            string = re.sub(pattern, repl, string)
        return string

    pattern = _compile_literal_replacement_pattern(tuple(replace_dict.keys()))
    return pattern.sub(lambda m: replace_dict[m.group(0)], string)


@lru_cache(maxsize=256)
def _compile_literal_replacement_pattern(keys: tuple[str, ...]) -> re.Pattern:
    # Preserve key order to avoid changing match precedence vs. the original implementation.
    return re.compile("|".join(re.escape(k) for k in keys))


def apply_transliterations(
    string: str,
    transliterations: Union[str, Sequence[str]],
    do_transliterate: bool,
    warn_unhandled_unicode: bool = True,
) -> str:
    """
    Apply transliteration steps in order.
    """
    if not do_transliterate:
        return string

    steps = _parse_transliteration_steps(transliterations)
    for step in steps:
        if step == "latin":
            string = unicodedata.normalize("NFKD", string)
        elif step == "ascii":
            string = unicodedata.normalize("NFKD", string).encode("ascii", "ignore").decode("utf-8", "ignore")
        elif step == "greek":
            string = string.translate(GREEK_TRANSLIT_TABLE)
        elif step == "cyrillic":
            if unidecode is not None:
                string = unidecode(string)
            else:
                logger.warning("unidecode module is not available for Cyrillic transliteration.")
        else:
            logger.warning("Unknown transliteration step: %s", step)

    if warn_unhandled_unicode and any(not c.isascii() for c in string):
        # Warn once with a compact list of remaining non-ASCII characters.
        remaining = sorted({c for c in string if not c.isascii()})
        logger.warning("Some characters were not transliterated or removed: %r", "".join(remaining))

    return string


def sanitize_string(string: str, sep_in: Union[str, Sequence[str], None], sep_out: str, parsing_option: int) -> str:
    """
    parsing_option=1 (preserved):
      - trim leading whitespace
      - remove leading separators
      - collapse separators into sep_out

    sep_in is more flexible:
      - r"\\." works as before
      - "." works literally
      - ["::", ".", "/"] works (multi-char supported)
      - None defaults to dot behavior
    """
    if parsing_option != 1:
        return string

    sep_pattern = _build_separator_regex(sep_in)

    string = re.sub(r"^\s*", "", string)
    string = re.sub(rf"^(?:{sep_pattern})+", "", string)
    string = re.sub(rf"(?:{sep_pattern})+", sep_out, string)
    return string


def handle_numerals(string: str, numerals_option: str) -> str:
    """
    Numeral handling:
      - 'asis': no changes (except make_valid_names may spell if entire string is digits)
      - 'remove': remove digit runs
      - 'spell': replace digit runs with num2words
    """
    if numerals_option == "remove":
        result = re.sub(r"\d+", "", string)
        result = re.sub(r"_$", "", result)
        result = re.sub(r"\s+$", "", result)
        return result

    if numerals_option == "spell":
        num2words = _get_num2words()
        return re.sub(r"\d+", lambda m: num2words(int(m.group(0))).replace(" ", "_"), string)

    return string


def make_valid_names(string: str, numerals: str) -> str:
    """
    Preserved behavior:
      - If entire string is digits and numerals=='asis', spell using num2words
      - Strip leading non-alphanumeric
      - Replace whitespace runs with underscore
    """
    if numerals == "asis" and string.isdigit():
        num2words = _get_num2words()
        string = num2words(int(string))

    string = re.sub(r"^[^a-zA-Z0-9]+", "", string)
    return re.sub(r"\s+", "_", string)


def apply_case_conversion(string: str, case: str, segment_sep: str = ".") -> str:
    """
    Convert to requested case, applied per segment separated by segment_sep.
    This keeps segment semantics intact (important for duplicate-handling).
    """
    segments = string.split(segment_sep) if segment_sep else [string]
    converted: List[str] = []

    for seg in segments:
        if case == "snake":
            converted.append(_to_snake(seg))
        elif case == "camel":
            s = _to_snake(seg)
            converted.append((s[0].lower() + s.title().replace("_", "")[1:]) if s else s)
        elif case == "pascal":
            s = _to_snake(seg)
            converted.append(s.title().replace("_", ""))
        elif case == "kebab":
            converted.append(seg.lower().replace("_", "-"))
        elif case == "space":
            converted.append(seg.lower().replace("_", " "))
        elif case == "title":
            converted.append(seg.replace("_", " ").title().replace(" ", "_"))
        elif case == "upper":
            converted.append(seg.upper())
        elif case == "lower":
            converted.append(seg.lower())
        elif case == "sentence":
            converted.append(seg.replace("_", " ").capitalize().replace(" ", "_"))
        else:
            raise ValueError(f"Unknown case style: {case}")

    return segment_sep.join(converted) if segment_sep else "".join(converted)


def remove_duplicate_names(names: List[str]) -> List[str]:
    """
    Preserve existing behavior: append counters to duplicates.
    Example: ["test","test","test"] -> ["test","test_2","test_3"]
    """
    dupe_count: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        if name in dupe_count:
            dupe_count[name] += 1
            out.append(f"{name}_{dupe_count[name] + 1}")
        else:
            dupe_count[name] = 0
            out.append(name)
    return out


def warn_about_micro_symbol(string: str, replace: Mapping[str, str]) -> None:
    """
    Preserve warning behavior for µ/μ if not accounted for in replacement rules.
    """
    for char in ("µ", "μ"):
        if char in string and not any(char in key for key in replace.keys()):
            logger.warning(
                "The character (%s) is present in the string. Consider adding a replacement rule if necessary.", char
            )


# ----------------------------- Utilities -----------------------------

def _to_snake(s: str) -> str:
    """
    Use inflection.underscore if available; otherwise conservative fallback.
    """
    if _inflection_underscore is not None:
        return _inflection_underscore(s)

    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"__+", "_", s)
    return s.lower()


def _normalize_case(case: str) -> str:
    key = (case or "").strip().lower()
    return _CASE_ALIASES.get(key, case)


def _normalize_numerals(numerals: str) -> str:
    key = (numerals or "").strip().lower()
    return _NUMERAL_ALIASES.get(key, numerals)


def _normalize_parsing_option(parsing_option: Union[int, str]) -> int:
    if isinstance(parsing_option, int):
        return parsing_option
    s = str(parsing_option).strip().lower()
    if s.isdigit():
        return int(s)
    return 1


def _parse_transliteration_steps(spec: Union[str, Sequence[str]]) -> List[str]:
    """
    Accepts:
      - "Latin-ASCII"
      - "latin ascii"
      - "Greek, ASCII"
      - ["Greek", "ASCII"]
    """
    if isinstance(spec, (list, tuple)):
        parts = [str(x) for x in spec]
    else:
        parts = re.split(r"[\s,|+;]+", str(spec).replace("-", " "))

    steps: List[str] = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        if p in ("nfkd", "normalize", "latin"):
            steps.append("latin")
        elif p == "ascii":
            steps.append("ascii")
        elif p in ("greek", "el"):
            steps.append("greek")
        elif p in ("cyrillic", "ru", "uk", "bg"):
            steps.append("cyrillic")
        else:
            steps.append(p)
    return steps


def _build_separator_regex(sep_in: Union[str, Sequence[str], None]) -> str:
    """
    Return a regex that matches ONE separator token.
    Caller can wrap with (?:...)+ to collapse runs.

    Compatibility and flexibility:
      - sep_in=r"\\." treated like the prior char-class fragment behavior (matches '.')
      - sep_in="." treated literally (matches '.')
      - sep_in=["::", ".", "/"] supports multi-char tokens
      - sep_in=None defaults to dot behavior
    """
    if sep_in is None:
        sep_in = r"\."

    if isinstance(sep_in, str):
        raw = sep_in
        # Backward-compatible: old usage passed fragments like r"\."
        if "\\" in raw:
            return rf"[{raw}]"
        # New convenience: literal characters set
        return rf"[{re.escape(raw)}]"

    seps = [str(x) for x in sep_in if str(x)]
    if not seps:
        return r"[\.]"

    if all(len(s) == 1 for s in seps):
        return rf"[{''.join(re.escape(s) for s in seps)}]"

    seps_sorted = sorted(seps, key=len, reverse=True)
    return rf"(?:{'|'.join(re.escape(s) for s in seps_sorted)})"


def _get_num2words():
    try:
        from num2words import num2words  # type: ignore
    except ImportError as e:
        logger.error("num2words module is required for numeral conversion (numerals='spell' or all-digit input).")
        raise e
    return num2words

'''

# UAT harness (Cell 2)
# Assumes Cell 1 defines:
#   - clean_names(...)
#   - clean_names_table(...)
#   - logger (module-level logger used by the implementation)
#
# This harness:
#   - Validates key behaviors and ergonomics
#   - Handles optional dependencies (num2words, unidecode) without crashing the run
#   - Optionally suppresses implementation log output during the run for clean UAT output

from typing import Any, Callable, Dict, List
import time
import logging
from contextlib import contextmanager


def _dep_available(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


@contextmanager
def _suppress_logger_output(target: logging.Logger):
    """
    Temporarily suppress all handler output from `target` logger without changing its configuration permanently.
    This prevents implementation warnings/errors from cluttering UAT output.
    """
    original_level = target.level
    original_disabled = target.disabled
    original_propagate = target.propagate

    # Capture and detach handlers to suppress emissions safely
    original_handlers = list(target.handlers)
    try:
        target.disabled = False
        target.propagate = False
        target.handlers = []
        target.setLevel(logging.CRITICAL + 1)  # above CRITICAL
        yield
    finally:
        target.handlers = original_handlers
        target.setLevel(original_level)
        target.disabled = original_disabled
        target.propagate = original_propagate


def run_uat(verbose: bool = True, suppress_impl_logs: bool = True) -> Dict[str, Any]:
    """
    Runs a comprehensive UAT suite against:
      - clean_names(...)
      - clean_names_table(...)

    Returns:
      {
        "ok": bool,
        "summary": {"passed": int, "failed": int, "total": int},
        "results": [ {name, ok, details, actual, expected}, ... ]
      }
    """
    results: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, details: str = "", actual: Any = None, expected: Any = None):
        results.append({
            "name": name,
            "ok": ok,
            "details": details,
            "actual": actual,
            "expected": expected,
        })
        if verbose:
            status = "PASS" if ok else "FAIL"
            msg = f"[{status}] {name}"
            if not ok:
                msg += f" | {details}"
            print(msg)

    def check_eq(name: str, actual: Any, expected: Any):
        ok = (actual == expected)
        record(name, ok, details="value mismatch" if not ok else "", actual=actual, expected=expected)

    def check_true(name: str, cond: bool, details: str = ""):
        record(name, bool(cond), details=details)

    def check_raises(name: str, fn: Callable[[], Any], exc_type: type):
        try:
            fn()
            record(name, False, details=f"expected exception {exc_type.__name__} but none was raised")
        except exc_type:
            record(name, True)
        except Exception as e:
            record(name, False, details=f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")

    # Locate implementation logger if present
    impl_logger = globals().get("logger", None)
    log_ctx = _suppress_logger_output(impl_logger) if (suppress_impl_logs and isinstance(impl_logger, logging.Logger)) else None

    try:
        if log_ctx:
            log_ctx.__enter__()

        # ---------- Sanity: functions exist ----------
        check_true("clean_names exists", callable(globals().get("clean_names")), "clean_names not found in globals")
        check_true("clean_names_table exists", callable(globals().get("clean_names_table")), "clean_names_table not found in globals")
        if not callable(globals().get("clean_names")) or not callable(globals().get("clean_names_table")):
            return {"ok": False, "results": results, "summary": {"passed": 0, "failed": len(results), "total": len(results)}}

        has_num2words = _dep_available("num2words")
        has_unidecode = _dep_available("unidecode")

        # ---------- Core behaviors ----------
        check_eq("snake basic", clean_names("This is a Test", case="snake"), "this_is_a_test")
        check_eq("camel basic", clean_names("This is a Test", case="camel"), "thisIsATest")
        check_eq("pascal basic", clean_names("This is a Test", case="pascal"), "ThisIsATest")
        check_eq("kebab basic", clean_names("This is a Test", case="kebab"), "this-is-a-test")
        check_eq("space basic", clean_names("This is a Test", case="space"), "this is a test")
        check_eq("title basic", clean_names("this is a test", case="title"), "This_Is_A_Test")
        check_eq("upper basic", clean_names("this is a test", case="upper"), "THIS_IS_A_TEST")
        check_eq("lower basic", clean_names("This IS a TEST", case="lower"), "this_is_a_test")
        check_eq("sentence basic", clean_names("this is a test", case="sentence"), "This_is_a_test")

        # Replacement behavior
        check_eq("replace defaults percent/number", clean_names("100% of #1"), "100_percent_of_number_1")
        check_eq("replace custom", clean_names("100% of #1", replace={"%": "percent", "#": "number"}), "100percent_of_number1")

        # Numerals behavior
        check_eq("numerals asis", clean_names("Chapter 10", numerals="asis"), "chapter_10")
        check_eq("numerals remove", clean_names("Chapter 10", numerals="remove"), "chapter")
        if has_num2words:
            check_eq("numerals spell", clean_names("Chapter 10", numerals="spell"), "chapter_ten")
        else:
            check_raises("numerals spell requires num2words", lambda: clean_names("Chapter 10", numerals="spell"), ModuleNotFoundError)

        # Duplicate handling
        check_eq("dupes in segments", clean_names("test.test.test", allow_dupes=False), "test.test_2.test_3")

        # Input validation
        check_raises("list input raises", lambda: clean_names(["a", "b"]), ValueError)

        # ---------- old_janitor branch ----------
        check_eq("old_janitor basic", clean_names("A.B.% #", case="old_janitor"), "a_b_percent_number")

        # ---------- Transliteration coverage ----------
        check_eq("Greek transliteration", clean_names("καλημέρα", transliterations="Greek", ascii=True), "kalhmera")
        if has_unidecode:
            out = clean_names("Привет мир", transliterations="Cyrillic ASCII", ascii=True)
            check_true("Cyrillic transliteration yields ASCII only", all(ord(c) < 128 for c in out), f"non-ASCII remained: {out!r}")
        else:
            check_true("Cyrillic transliteration skipped (unidecode missing)", True)

        # ---------- Ease-of-typing / flexibility ----------
        check_eq("case alias snake_case", clean_names("This is a Test", case="snake_case"), "this_is_a_test")
        check_eq("numerals alias keep", clean_names("Chapter 10", numerals="keep"), "chapter_10")
        check_eq("sep_in literal '.' convenience", clean_names("test..test", sep_in="."), "test.test_2")
        check_eq("sep_in list convenience", clean_names("a/b.c", sep_in=["/", "."], sep_out="."), "a.b.c")
        check_eq("multi-char sep_in tokens", clean_names("a::b::c", sep_in=["::"], sep_out="."), "a.b.c")
        check_eq("sep_out custom", clean_names("a/b/c", sep_in="/", sep_out="__"), "a__b__c")

        out = clean_names("καλημέρα", transliterations="Greek", ascii=False, transliterate=True)
        check_eq("transliterate override (ascii False but transliterate True)", out, "kalhmera")

        out = clean_names("abc123def", replace={r"\d+": "_NUM_"}, replace_regex=True)
        check_eq("replace_regex replaces digit runs", out, "abc_num_def")

        # ---------- Data table mapping ----------
        mapping = clean_names_table(["This is a Test", "Chapter 10", "test.test.test"], case="snake")
        check_eq("mapping row count", len(mapping), 3)
        check_eq("mapping row 1", mapping[0], ("This is a Test", "this_is_a_test"))
        check_eq("mapping row 2", mapping[1], ("Chapter 10", "chapter_10"))
        check_eq("mapping row 3", mapping[2], ("test.test.test", "test.test_2.test_3"))

        # Digits-only values require num2words when numerals='asis'
        if has_num2words:
            mapping2 = clean_names_table([None, 123, "naïve café"], skip_none=False)
            check_eq("mapping2 keeps None row", mapping2[0][0], "")
            check_true("mapping2 int original is stringified", mapping2[1][0] == "123", f"got {mapping2[1][0]!r}")
            check_true("mapping2 int rename non-empty", isinstance(mapping2[1][1], str) and len(mapping2[1][1]) > 0, "rename empty for digits-only")
            check_true("mapping2 int rename has no spaces", " " not in mapping2[1][1], f"spaces in {mapping2[1][1]!r}")
            check_eq("mapping2 ascii removes accents", mapping2[2][1], "naive_cafe")
        else:
            check_raises("digits-only requires num2words (clean_names)", lambda: clean_names("123"), ModuleNotFoundError)
            check_raises("digits-only requires num2words (clean_names_table)", lambda: clean_names_table([123]), ModuleNotFoundError)

            mapping2 = clean_names_table([None, "123x", "naïve café"], skip_none=False)
            check_eq("mapping2 safe keeps None row", mapping2[0][0], "")
            check_eq("mapping2 safe '123x' ok", mapping2[1], ("123x", "123x"))
            check_eq("mapping2 safe ascii removes accents", mapping2[2][1], "naive_cafe")

        mapping3 = clean_names_table([None, "x"], skip_none=True)
        check_eq("mapping skip_none drops None", mapping3, [("x", "x")])

        # ---------- Output quality checks ----------
        out = clean_names("Hello World", case="snake")
        check_true("snake has no spaces", " " not in out, f"unexpected spaces in {out!r}")

        out = clean_names("naïve café", ascii=True)
        check_true("ascii=True yields ASCII only", all(ord(c) < 128 for c in out), f"non-ASCII remained: {out!r}")

        out1 = clean_names("x.x.x", allow_dupes=False)
        out2 = clean_names("x.x.x", allow_dupes=False)
        check_eq("deterministic output", out1, out2)

        # ---------- Basic performance smoke test ----------
        t0 = time.time()
        for _ in range(5000):
            clean_names("This is a Test 123", case="snake", numerals="asis")
        dt = time.time() - t0
        check_true("performance smoke test (5000 ops < 2.0s)", dt < 2.0, f"took {dt:.3f}s")

        # ---------- Summary ----------
        passed = sum(1 for r in results if r["ok"])
        failed = sum(1 for r in results if not r["ok"])
        summary = {"passed": passed, "failed": failed, "total": len(results)}

        if verbose:
            print("\n--- UAT SUMMARY ---")
            print(f"Passed: {passed} | Failed: {failed} | Total: {len(results)}")
            if failed:
                print("\nFailures:")
                for r in results:
                    if not r["ok"]:
                        print(f"- {r['name']}: {r['details']} | actual={r['actual']!r} expected={r['expected']!r}")

        return {"ok": failed == 0, "summary": summary, "results": results}

    finally:
        if log_ctx:
            log_ctx.__exit__(None, None, None)


# Execute UAT
uat_report = run_uat(verbose=True, suppress_impl_logs=True)
uat_report


'''