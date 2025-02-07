import re
import unidecode
import unicodedata
from inflection import underscore as snakecase
from num2words import num2words
from typing import Dict, Optional

GREEK_TRANSLIT_RULES = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "h", "θ": "th", 
    "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", 
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u", "φ": "f", "χ": "ch", "ψ": "ps", 
    "ω": "o", "ά": "a", "έ": "e", "ί": "i", "ό": "o", "ύ": "u", "ή": "h", "ώ": "o", 
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "H", "Θ": "TH", 
    "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P", 
    "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O"
}

def clean_names(
    string: str,
    case: str = "snake",
    replace: Optional[Dict[str, str]] = None,
    ascii: bool = True,
    use_make_names: bool = True,
    allow_dupes: bool = False,
    sep_in: str = "\\.",
    transliterations: str = "Latin-ASCII",
    parsing_option: int = 1,
    numerals: str = "asis"
) -> str:
    """
    Cleans a given string according to specified formatting and transliteration options.
    """
    if not isinstance(string, str):
        raise ValueError("`string` must be a single string, not a list or other type.")

    replace_dict = replace or {"'": "", '"': "", "%": "_percent_", "#": "_number_"}
    warn_about_micro_symbol(string, replace_dict)

    string = apply_replacements(string, replace_dict)
    string = apply_transliterations(string, transliterations, ascii)
    string = sanitize_string(string, sep_in, parsing_option)

    if numerals != "asis":
        string = handle_numerals(string, numerals)

    if use_make_names:
        string = make_valid_names(string)

    string = apply_case_conversion(string, case)

    if not allow_dupes:
        string = '.'.join(remove_duplicate_names(string.split('.')))

    return string

def apply_replacements(string: str, replace_dict: Dict[str, str]) -> str:
    pattern = "|".join(map(re.escape, replace_dict.keys()))
    return re.sub(pattern, lambda m: replace_dict[m.group(0)], string)

def apply_transliterations(string: str, transliterations: str, ascii: bool) -> str:
    if ascii:
        for transliteration in transliterations.split('-'):
            if transliteration == "Latin":
                string = unicodedata.normalize('NFKD', string)
            elif transliteration == "ASCII":
                string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            elif transliteration == "Greek":
                pattern = "|".join(map(re.escape, GREEK_TRANSLIT_RULES.keys()))
                string = re.sub(pattern, lambda m: GREEK_TRANSLIT_RULES[m.group(0)], string)
            elif transliteration == "Cyrillic":
                string = unidecode(string)
    return string

def sanitize_string(string: str, sep_in: str, parsing_option: int) -> str:
    if parsing_option == 1:
        string = re.sub(rf"^\s*[{sep_in}]*", "", string)
        string = re.sub(rf"[{sep_in}]+", ".", string)
    return string

def handle_numerals(string: str, numerals_option: str) -> str:
    if numerals_option == "remove":
        return re.sub(r"\d+", "", string).strip()
    elif numerals_option == "spell":
        return re.sub(r"\d+", lambda x: num2words(int(x.group(0))).replace(" ", "_"), string)
    return string

def make_valid_names(string: str) -> str:
    string = re.sub(r"^[^a-zA-Z0-9]+", "", string)
    return re.sub(r"\s+", "_", string)

def apply_case_conversion(string: str, case: str) -> str:
    cases = {
        "snake": lambda s: snakecase(s),
        "camel": lambda s: s[0].lower() + snakecase(s).title().replace("_", "")[1:],
        "pascal": lambda s: snakecase(s).title().replace("_", ""),
        "kebab": lambda s: s.lower().replace("_", "-"),
        "space": lambda s: s.lower().replace("_", " "),
        "title": lambda s: s.replace('_', ' ').title().replace(' ', '_'),
        "upper": lambda s: s.upper(),
        "lower": lambda s: s.lower(),
        "sentence": lambda s: s.capitalize().replace('_', ' ').capitalize().replace(' ', '_')
    }
    return cases.get(case, lambda s: s)(string)

def remove_duplicate_names(names: list) -> list:
    dupe_count, new_names = {}, []
    for name in names:
        if name in dupe_count:
            dupe_count[name] += 1
            new_names.append(f"{name}_{dupe_count[name] + 1}")
        else:
            dupe_count[name] = 0
            new_names.append(name)
    return new_names

def warn_about_micro_symbol(string: str, replace: Dict[str, str]):
    special_characters = ["µ", "μ"]
    for char in special_characters:
        if char in string and char not in replace:
            print(f"Warning: The character ({char}) is present in the string. Consider adding a replacement rule.")
