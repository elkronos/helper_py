import re
import unicodedata
import logging
from typing import Union, List, Dict, Optional

from inflection import underscore as snakecase

# Try to import unidecode if available (used for Cyrillic transliteration)
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

# Set up logging for debug messages.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)

# Dictionary for Greek transliteration rules.
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
    "Ύ": "Y", "Ή": "H", "Ώ": "O"
}


def clean_names(string: str,
                case: str = "snake",
                replace: Optional[Dict[str, str]] = None,
                ascii: bool = True,
                use_make_names: bool = True,
                allow_dupes: bool = False,
                sep_in: str = "\\.",
                transliterations: str = "Latin-ASCII",
                parsing_option: int = 1,
                numerals: str = "asis") -> str:
    """
    Clean and standardize a given string into a valid name.

    Parameters:
        string (str): The input string.
        case (str): The desired case formatting (e.g. 'snake', 'camel', 'pascal', 'kebab', etc.).
        replace (Optional[Dict[str, str]]): A dictionary of replacement rules.
        ascii (bool): Whether to force ASCII conversion.
        use_make_names (bool): Whether to apply further name validation.
        allow_dupes (bool): Whether to allow duplicate segments.
        sep_in (str): The regex string representing input separator(s).
        transliterations (str): A hyphen-separated list of transliteration methods.
        parsing_option (int): Option for sanitization (currently supports option 1).
        numerals (str): Option for numeral handling ('asis', 'remove', 'spell').

    Returns:
        str: The cleaned string.
    """
    if isinstance(string, list):
        raise ValueError("`string` must not be a list, it should be a single string.")
    logger.debug("Initial string: %s", string)
    logger.debug("Parameters - case: %s, replace: %s, ascii: %s, use_make_names: %s, "
                 "allow_dupes: %s, sep_in: %s, transliterations: %s, parsing_option: %s, numerals: %s",
                 case, replace, ascii, use_make_names, allow_dupes, sep_in, transliterations, parsing_option, numerals)

    # Special branch for "old_janitor" style.
    if case == "old_janitor":
        replace_dict = {"'": "", '"': "", "%": "percent", "#": "number"}
        warn_about_micro_symbol(string, replace_dict)
        logger.debug("Replacements (old_janitor): %s", replace_dict)
        cleaned_str = apply_replacements(string, replace_dict)
        logger.debug("After replacements (old_janitor): %s", cleaned_str)
        cleaned_str = make_valid_names(cleaned_str, numerals)
        logger.debug("After making valid names (old_janitor): %s", cleaned_str)
        cleaned_str = re.sub(r"[.]+", "_", cleaned_str)
        cleaned_str = re.sub(r"[_]+", "_", cleaned_str)
        cleaned_str = cleaned_str.lower()
        cleaned_str = re.sub(r"_$", "", cleaned_str)
        if not allow_dupes:
            parts = cleaned_str.split('_')
            cleaned_str = '_'.join(remove_duplicate_names(parts))
        logger.debug("Final string (old_janitor): %s", cleaned_str)
        return cleaned_str

    # Use provided replacement rules or fall back to defaults.
    replace_dict = replace if replace is not None else {"'": "", '"': "", "%": "_percent_", "#": "_number_"}
    warn_about_micro_symbol(string, replace_dict)
    logger.debug("Replacements: %s", replace_dict)

    cleaned_str = apply_replacements(string, replace_dict)
    logger.debug("After replacements: %s", cleaned_str)
    cleaned_str = apply_transliterations(cleaned_str, transliterations, ascii)
    logger.debug("After transliterations: %s", cleaned_str)
    cleaned_str = sanitize_string(cleaned_str, sep_in, parsing_option)
    logger.debug("After sanitization: %s", cleaned_str)

    if numerals != "asis":
        cleaned_str = handle_numerals(cleaned_str, numerals)
        logger.debug("After handling numerals: %s", cleaned_str)

    if use_make_names:
        cleaned_str = make_valid_names(cleaned_str, numerals)
        logger.debug("After making valid names: %s", cleaned_str)

    cleaned_str = apply_case_conversion(cleaned_str, case)
    logger.debug("After case conversion: %s", cleaned_str)

    if not allow_dupes:
        # Split on period (the default output separator from sanitization)
        parts = cleaned_str.split('.')
        cleaned_str = '.'.join(remove_duplicate_names(parts))
        logger.debug("After removing duplicates: %s", cleaned_str)

    logger.debug("Final cleaned string: %s", cleaned_str)
    return cleaned_str


def apply_replacements(string: str, replace_dict: Dict[str, str]) -> str:
    """
    Apply replacement rules to the input string.

    Parameters:
        string (str): The string to modify.
        replace_dict (Dict[str, str]): A dictionary of patterns and their replacements.

    Returns:
        str: The modified string.
    """
    logger.debug("apply_replacements input: %s", string)
    pattern = "|".join(map(re.escape, replace_dict.keys()))
    result = re.sub(pattern, lambda m: replace_dict[m.group(0)], string)
    logger.debug("apply_replacements output: %s", result)
    return result


def apply_transliterations(string: str, transliterations: str, force_ascii: bool) -> str:
    """
    Apply transliterations to the string if force_ascii is True.

    Parameters:
        string (str): The string to transliterate.
        transliterations (str): Hyphen-separated list of transliteration methods.
        force_ascii (bool): Whether to enforce ASCII conversion.

    Returns:
        str: The transliterated string.
    """
    logger.debug("apply_transliterations input: %s, transliterations: %s, force_ascii: %s",
                 string, transliterations, force_ascii)
    if force_ascii:
        for transliteration in transliterations.split('-'):
            if transliteration == "Latin":
                string = unicodedata.normalize('NFKD', string)
            elif transliteration == "ASCII":
                string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            elif transliteration == "Greek":
                pattern = "|".join(map(re.escape, GREEK_TRANSLIT_RULES.keys()))
                string = re.sub(pattern, lambda m: GREEK_TRANSLIT_RULES[m.group(0)], string)
            elif transliteration == "Cyrillic":
                if unidecode:
                    string = unidecode(string)
                else:
                    logger.warning("unidecode module is not available for Cyrillic transliteration.")
            # Warn about any non-ASCII characters that remain.
            for char in string:
                if not char.isascii() and char not in GREEK_TRANSLIT_RULES:
                    logger.warning("The character %s has not been transliterated or replaced.", char)
    logger.debug("apply_transliterations output: %s", string)
    return string


def sanitize_string(string: str, sep_in: str, parsing_option: int) -> str:
    """
    Sanitize the string based on the input separator and parsing option.

    Parameters:
        string (str): The string to sanitize.
        sep_in (str): The regex string for input separators.
        parsing_option (int): The parsing option (currently supports 1).

    Returns:
        str: The sanitized string.
    """
    logger.debug("sanitize_string input: %s, sep_in: %s, parsing_option: %s", string, sep_in, parsing_option)
    if parsing_option == 1:
        string = re.sub(rf"^\s*[{sep_in}]*", "", string)
        string = re.sub(rf"[{sep_in}]+", ".", string)
    logger.debug("sanitize_string output: %s", string)
    return string


def handle_numerals(string: str, numerals_option: str) -> str:
    """
    Handle numerals in the string according to numerals_option.

    Parameters:
        string (str): The string containing numerals.
        numerals_option (str): How to handle numerals ('asis', 'remove', 'spell').

    Returns:
        str: The string with numerals handled.
    """
    logger.debug("handle_numerals input: %s, numerals_option: %s", string, numerals_option)
    if numerals_option == "remove":
        result = re.sub(r"\d+", "", string)
        result = re.sub(r"_$", "", result)  # Remove trailing underscore if any.
        result = re.sub(r"\s+$", "", result)  # Remove trailing spaces if any.
        logger.debug("handle_numerals (remove) output: %s", result)
        return result
    elif numerals_option == "spell":
        try:
            from num2words import num2words
        except ImportError:
            logger.error("num2words module is required for spelling numerals.")
            raise
        result = re.sub(r"\d+", lambda x: num2words(int(x.group(0))).replace(" ", "_"), string)
        logger.debug("handle_numerals (spell) output: %s", result)
        return result
    logger.debug("handle_numerals (asis) output: %s", string)
    return string


def make_valid_names(string: str, numerals: str) -> str:
    """
    Ensure the string is a valid name by removing invalid starting characters
    and replacing spaces with underscores. If the entire string is numeric and
    numerals is 'asis', convert it to its word form.

    Parameters:
        string (str): The string to validate.
        numerals (str): Numeral handling mode.

    Returns:
        str: A valid name string.
    """
    logger.debug("make_valid_names input: %s", string)
    if numerals == "asis" and string.isdigit():
        try:
            from num2words import num2words
        except ImportError:
            logger.error("num2words module is required for numeral conversion.")
            raise
        string = num2words(int(string))
    string = re.sub(r"^[^a-zA-Z0-9]+", "", string)
    result = re.sub(r"\s+", "_", string)
    logger.debug("make_valid_names output: %s", result)
    return result


def apply_case_conversion(string: str, case: str) -> str:
    """
    Convert the string to the specified case style.

    Parameters:
        string (str): The string to convert.
        case (str): The target case style (e.g., 'snake', 'camel', 'pascal', 'kebab', etc.).

    Returns:
        str: The case-converted string.

    Raises:
        ValueError: If the case style is unknown.
    """
    logger.debug("apply_case_conversion input: %s, case: %s", string, case)
    if case == "snake":
        result = snakecase(string)
    elif case == "camel":
        s = snakecase(string)
        result = s[0].lower() + s.title().replace("_", "")[1:] if s else s
    elif case == "pascal":
        s = snakecase(string)
        result = s.title().replace("_", "")
    elif case == "kebab":
        result = string.lower().replace("_", "-")
    elif case == "space":
        result = string.lower().replace("_", " ")
    elif case == "title":
        result = string.replace('_', ' ').title().replace(' ', '_')
    elif case == "upper":
        result = string.upper()
    elif case == "lower":
        result = string.lower()
    elif case == "sentence":
        result = string.capitalize().replace('_', ' ').capitalize().replace(' ', '_')
    else:
        raise ValueError(f"Unknown case style: {case}")
    logger.debug("apply_case_conversion output (%s): %s", case, result)
    return result


def remove_duplicate_names(names: List[str]) -> List[str]:
    """
    Remove duplicate name segments by appending a counter to subsequent duplicates.

    Parameters:
        names (List[str]): A list of name segments.

    Returns:
        List[str]: The list with duplicate segments modified.
    """
    logger.debug("remove_duplicate_names input: %s", names)
    dupe_count: Dict[str, int] = {}
    new_names: List[str] = []
    for name in names:
        if name in dupe_count:
            dupe_count[name] += 1
            new_names.append(f"{name}_{dupe_count[name] + 1}")
        else:
            dupe_count[name] = 0
            new_names.append(name)
    logger.debug("remove_duplicate_names output: %s", new_names)
    return new_names


def warn_about_micro_symbol(string: str, replace: Dict[str, str]) -> None:
    """
    Warn if micro symbols (µ, μ) are present in the string and not accounted for in the replacement rules.

    Parameters:
        string (str): The input string.
        replace (Dict[str, str]): The dictionary of replacement rules.
    """
    special_characters = ["µ", "μ"]
    for char in special_characters:
        if char in string and not any(char in key for key in replace.keys()):
            logger.warning("The character (%s) is present in the string. Consider adding a replacement rule if necessary.", char)


# ================== Unit Tests ==================

import unittest

class TestCleanNamesFunction(unittest.TestCase):

    def test_snake_case_conversion(self):
        self.assertEqual(clean_names("This is a Test", case="snake"), "this_is_a_test")

    def test_camel_case_conversion(self):
        self.assertEqual(clean_names("This is a Test", case="camel"), "thisIsATest")

    def test_pascal_case_conversion(self):
        self.assertEqual(clean_names("This is a Test", case="pascal"), "ThisIsATest")

    def test_kebab_case_conversion(self):
        self.assertEqual(clean_names("This is a Test", case="kebab"), "this-is-a-test")

    def test_space_case_conversion(self):
        self.assertEqual(clean_names("This is a Test", case="space"), "this is a test")

    def test_title_case_conversion(self):
        self.assertEqual(clean_names("this is a test", case="title"), "This_Is_A_Test")

    def test_upper_case_conversion(self):
        self.assertEqual(clean_names("this is a test", case="upper"), "THIS_IS_A_TEST")

    def test_lower_case_conversion(self):
        self.assertEqual(clean_names("This IS a TEST", case="lower"), "this_is_a_test")

    def test_sentence_case_conversion(self):
        self.assertEqual(clean_names("this is a test", case="sentence"), "This_is_a_test")

    def test_transliteration(self):
        # Transliterate Greek text (ensure ascii conversion is active).
        self.assertEqual(clean_names("καλημέρα", transliterations="Greek", ascii=True), "kalhmera")

    def test_replace_characters(self):
        self.assertEqual(clean_names("100% of #1", replace={"%": "percent", "#": "number"}), "100percent_of_number1")

    def test_ascii_conversion(self):
        self.assertEqual(clean_names("naïve café", ascii=True), "naive_cafe")

    def test_handle_numerals_as_is(self):
        self.assertEqual(clean_names("Chapter 10", numerals="asis"), "chapter_10")

    def test_handle_numerals_remove(self):
        self.assertEqual(clean_names("Chapter 10", numerals="remove"), "chapter")

    def test_handle_numerals_spell(self):
        # Typically, "10" should be converted to "ten"
        self.assertEqual(clean_names("Chapter 10", numerals="spell"), "chapter_ten")

    def test_duplicate_names_removal(self):
        self.assertEqual(clean_names("test.test.test", allow_dupes=False), "test.test_2.test_3")


if __name__ == "__main__":
    unittest.main()
