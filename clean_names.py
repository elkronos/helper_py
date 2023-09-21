import re
import unidecode
import unicodedata  # Added this import
from snakecase import snakecase

# If you're going to use num2words, ensure it's installed:
# !pip install num2words

# Greek Transliteration Dictionary externalized
GREEK_TRANSLIT_RULES = {
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

# Main function
def clean_names(string, case="snake", replace=None, ascii=True, use_make_names=True, allow_dupes=False,
                sep_in="\\.", transliterations="Latin-ASCII", parsing_option=1, numerals="asis"):
     """
    Cleans and transforms the given string based on the specified parameters.

    Parameters:
    - string (str): The input string to be cleaned.
    - case (str): Desired case style. Supported styles: snake, camel, pascal, kebab, space, title, upper, lower, sentence, old_janitor.
    - replace (dict): Dictionary of character substitutions. E.g., {"%": "_percent_", "#": "_number_"}.
    - ascii (bool): If True, non-ASCII characters are removed or transliterated.
    - use_make_names (bool): If True, ensures the cleaned string is a valid variable name.
    - allow_dupes (bool): If False, removes duplicate name segments.
    - sep_in (str): Regular expression pattern for separators in the input string.
    - transliterations (str): Specifies transliteration rules. Supported rules: Latin, ASCII, Greek, Cyrillic.
    - parsing_option (int): Specifies parsing options. Currently only 1 is implemented.
    - numerals (str): How to handle numeric values. Supported options: asis, remove, spell.

    Returns:
    - str: The cleaned string.
    """

    if isinstance(string, list):
        raise ValueError("`string` must not be a list, it should be a single string.")

    if case == "old_janitor":
        replace_dict = {
            "'": "",
            '"': "",
            "%": "percent",
            "#": "_number_"
        }
        warn_about_micro_symbol(string, replace_dict)
        cleaned_str = apply_replacements(string, replace_dict)
        cleaned_str = make_valid_names(cleaned_str)
        cleaned_str = re.sub(r"[.]+", "_", cleaned_str)
        cleaned_str = re.sub(r"[_]+", "_", cleaned_str)
        cleaned_str = cleaned_str.lower()
        cleaned_str = re.sub(r"_$", "", cleaned_str)
        if not allow_dupes:
            cleaned_str = '.'.join(remove_duplicate_names(cleaned_str.split('.')))
        return cleaned_str
    
    replace_dict = replace or {
        "'": "",
        '"': "",
        "%": "_percent_",
        "#": "_number_"
    }
    
    warn_about_micro_symbol(string, replace_dict)
    
    cleaned_str = apply_replacements(string, replace_dict)
    cleaned_str = apply_transliterations(cleaned_str, transliterations, ascii)
    cleaned_str = sanitize_string(cleaned_str, sep_in, parsing_option)
    
    if numerals != "asis":
        cleaned_str = handle_numerals(cleaned_str, numerals)
    
    if use_make_names:
        cleaned_str = make_valid_names(cleaned_str)
        
    cleaned_str = apply_case_conversion(cleaned_str, case)
    
    if not allow_dupes:
        cleaned_str = '.'.join(remove_duplicate_names(cleaned_str.split('.')))

    return cleaned_str

# Helper functions
def apply_replacements(string, replace_dict):
    """
    Apply multiple character replacements to the input string based on a dictionary.

    Parameters:
    - string (str): The input string.
    - replace_dict (dict): Dictionary of character substitutions.

    Returns:
    - str: String with substitutions applied.
    """
    pattern = "|".join(map(re.escape, replace_dict.keys()))
    return re.sub(pattern, lambda m: replace_dict[m.group(0)], string)

def apply_transliterations(string, transliterations, ascii):
    """
    Transliterate characters in the input string based on specified rules.

    Parameters:
    - string (str): The input string.
    - transliterations (str): Transliteration rules separated by hyphens.
    - ascii (bool): If True, enforces ASCII-only transliterations.

    Returns:
    - str: Transliterated string.
    """
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

            # Warn about characters that haven't been transliterated or replaced
            for char in string:
                if not char.isascii() and char not in GREEK_TRANSLIT_RULES:
                    print(f"Warning: The character {char} has not been transliterated or replaced.")

    return string

def sanitize_string(string, sep_in, parsing_option):
    """
    Cleans the string by removing unwanted characters and separators.

    Parameters:
    - string (str): The input string.
    - sep_in (str): Regular expression pattern for separators in the input string.
    - parsing_option (int): Parsing option to apply.

    Returns:
    - str: Sanitized string.
    """
    if parsing_option == 1:
        string = re.sub(rf"^\s*[{sep_in}]*", "", string)
        string = re.sub(rf"[{sep_in}]+", ".", string)
    
    return string

def handle_numerals(string, numerals_option):
    """
    Process numerals in the string based on the specified option.

    Parameters:
    - string (str): The input string.
    - numerals_option (str): Specifies how numerals should be treated.

    Returns:
    - str: String with numerals processed.
    """
    if numerals_option == "remove":
        return re.sub(r"\d+", "", string)
    elif numerals_option == "spell":
        from num2words import num2words
        return re.sub(r"\d+", lambda x: num2words(int(x.group(0))), string)
    return string

def make_valid_names(string):
    """
    Ensures that the string is a valid name, e.g., for use as a variable or column name.

    Parameters:
    - string (str): The input string.

    Returns:
    - str: A valid name string.
    """
    # If the string is entirely numeric, convert it to its word form
    if string.isdigit():
        from num2words import num2words
        string = num2words(int(string))

    string = re.sub(r"^[^a-zA-Z]+", "", string)
    return re.sub(r"\s+", "_", string)

def apply_case_conversion(string, case):
    """
    Converts the string to the specified case style.

    Parameters:
    - string (str): The input string.
    - case (str): Desired case style.

    Returns:
    - str: String in the specified case style.
    """
    if case == "snake":
        return snakecase(string)
    elif case == "camel":
        s = snakecase(string)
        return s[0].lower() + s.title().replace("_", "")[1:]
    elif case == "pascal":
        s = snakecase(string)
        return s.title().replace("_", "")
    elif case == "kebab":
        return snakecase(string).replace("_", "-")
    elif case == "space":
        return snakecase(string).replace("_", " ")
    elif case == "title":
        return string.title()
    elif case == "upper":
        return string.upper()
    elif case == "lower":
        return string.lower()
    elif case == "sentence":
        return string.capitalize()
    else:
        raise ValueError(f"Unknown case style: {case}")
    
    return string

def remove_duplicate_names(names):
    """
    Remove duplicate name segments in a list by appending numbers.

    Parameters:
    - names (list of str): List of name segments.

    Returns:
    - list of str: List with duplicates removed or numbered.
    """
    dupe_count = {name: 0 for name in names}
    new_names = []
    for name in names:
        dupe_count[name] += 1
        if dupe_count[name] > 1:
            new_names.append(f"{name}_{dupe_count[name]}")
        else:
            new_names.append(name)
    return new_names

def warn_about_micro_symbol(string, replace):
    """
    Issues a warning if the micro symbol (or other special characters) is present in the string 
    and not handled by the replacement dictionary.

    Parameters:
    - string (str): The input string.
    - replace (dict): Dictionary of character substitutions.
    """
    special_characters = ["µ", "μ"]
    for char in special_characters:
        if char in string and not any([char in key for key in replace.keys()]):
            print(f"Warning: The character ({char}) is present in the string. Consider adding a replacement rule if necessary.")