import re
import unidecode
import unicodedata
from inflection import underscore as snakecase

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

def clean_names(string, case="snake", replace=None, ascii=True, use_make_names=True, allow_dupes=False,
                sep_in="\\.", transliterations="Latin-ASCII", parsing_option=1, numerals="asis"):
    print(f"Initial string: {string}")
    print(f"Parameters - case: {case}, replace: {replace}, ascii: {ascii}, use_make_names: {use_make_names}, allow_dupes: {allow_dupes}, sep_in: {sep_in}, transliterations: {transliterations}, parsing_option: {parsing_option}, numerals: {numerals}")

    if isinstance(string, list):
        raise ValueError("`string` must not be a list, it should be a single string.")

    if case == "old_janitor":
        replace_dict = {
            "'": "",
            '"': "",
            "%": "percent",
            "#": "number"
        }
        warn_about_micro_symbol(string, replace_dict)
        print(f"Replacements (old_janitor): {replace_dict}")
        cleaned_str = apply_replacements(string, replace_dict)
        print(f"After replacements (old_janitor): {cleaned_str}")
        cleaned_str = make_valid_names(cleaned_str, numerals)
        print(f"After making valid names (old_janitor): {cleaned_str}")
        cleaned_str = re.sub(r"[.]+", "_", cleaned_str)
        cleaned_str = re.sub(r"[_]+", "_", cleaned_str)
        cleaned_str = cleaned_str.lower()
        cleaned_str = re.sub(r"_$", "", cleaned_str)
        if not allow_dupes:
            cleaned_str = '_'.join(remove_duplicate_names(cleaned_str.split('_')))
        print(f"Final string (old_janitor): {cleaned_str}")
        return cleaned_str
    
    replace_dict = replace or {
        "'": "",
        '"': "",
        "%": "_percent_",
        "#": "_number_"
    }
    
    warn_about_micro_symbol(string, replace_dict)
    print(f"Replacements: {replace_dict}")
    
    cleaned_str = apply_replacements(string, replace_dict)
    print(f"After replacements: {cleaned_str}")
    cleaned_str = apply_transliterations(cleaned_str, transliterations, ascii)
    print(f"After transliterations: {cleaned_str}")
    cleaned_str = sanitize_string(cleaned_str, sep_in, parsing_option)
    print(f"After sanitization: {cleaned_str}")
    
    if numerals != "asis":
        cleaned_str = handle_numerals(cleaned_str, numerals)
        print(f"After handling numerals: {cleaned_str}")
    
    if use_make_names:
        cleaned_str = make_valid_names(cleaned_str, numerals)
        print(f"After making valid names: {cleaned_str}")
        
    cleaned_str = apply_case_conversion(cleaned_str, case)
    print(f"After case conversion: {cleaned_str}")
    
    if not allow_dupes:
        cleaned_str = '.'.join(remove_duplicate_names(cleaned_str.split('.')))
        print(f"After removing duplicates: {cleaned_str}")

    print(f"Final cleaned string: {cleaned_str}")
    return cleaned_str

def apply_replacements(string, replace_dict):
    print(f"apply_replacements input: {string}")
    pattern = "|".join(map(re.escape, replace_dict.keys()))
    result = re.sub(pattern, lambda m: replace_dict[m.group(0)], string)
    print(f"apply_replacements output: {result}")
    return result

def apply_transliterations(string, transliterations, ascii):
    print(f"apply_transliterations input: {string}, transliterations: {transliterations}, ascii: {ascii}")
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
    print(f"apply_transliterations output: {string}")
    return string

def sanitize_string(string, sep_in, parsing_option):
    print(f"sanitize_string input: {string}, sep_in: {sep_in}, parsing_option: {parsing_option}")
    if parsing_option == 1:
        string = re.sub(rf"^\s*[{sep_in}]*", "", string)
        string = re.sub(rf"[{sep_in}]+", ".", string)
    print(f"sanitize_string output: {string}")
    return string

def handle_numerals(string, numerals_option):
    print(f"handle_numerals input: {string}, numerals_option: {numerals_option}")
    if numerals_option == "remove":
        result = re.sub(r"\d+", "", string)
        result = re.sub(r"_$", "", result)  # Remove trailing underscore if any
        result = re.sub(r"\s+$", "", result)  # Remove trailing spaces if any
        print(f"handle_numerals (remove) output: {result}")
        return result
    elif numerals_option == "spell":
        from num2words import num2words
        result = re.sub(r"\d+", lambda x: num2words(int(x.group(0))).replace(" ", "_"), string)
        print(f"handle_numerals (spell) output: {result}")
        return result
    print(f"handle_numerals (asis) output: {string}")
    return string

def make_valid_names(string, numerals):
    print(f"make_valid_names input: {string}")
    # If the string is entirely numeric, convert it to its word form
    if numerals == "asis" and string.isdigit():
        from num2words import num2words
        string = num2words(int(string))

    string = re.sub(r"^[^a-zA-Z0-9]+", "", string)
    result = re.sub(r"\s+", "_", string)
    print(f"make_valid_names output: {result}")
    return result

def apply_case_conversion(string, case):
    print(f"apply_case_conversion input: {string}, case: {case}")
    if case == "snake":
        result = snakecase(string)
    elif case == "camel":
        s = snakecase(string)
        result = s[0].lower() + s.title().replace("_", "")[1:]
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
    print(f"apply_case_conversion output ({case}): {result}")
    return result

def remove_duplicate_names(names):
    print(f"remove_duplicate_names input: {names}")
    dupe_count = {}
    new_names = []
    for name in names:
        if name in dupe_count:
            dupe_count[name] += 1
            new_names.append(f"{name}_{dupe_count[name] + 1}")
        else:
            dupe_count[name] = 0
            new_names.append(name)
    print(f"remove_duplicate_names output: {new_names}")
    return new_names

def warn_about_micro_symbol(string, replace):
    special_characters = ["µ", "μ"]
    for char in special_characters:
        if char in string and not any([char in key for key in replace.keys()]):
            print(f"Warning: The character ({char}) is present in the string. Consider adding a replacement rule if necessary.")

# Test script for clean_names function
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
        self.assertEqual(clean_names("καλημέρα", transliterations="Greek"), "kalhmera")

    def test_replace_characters(self):
        self.assertEqual(clean_names("100% of #1", replace={"%": "percent", "#": "number"}), "100percent_of_number1")

    def test_ascii_conversion(self):
        self.assertEqual(clean_names("naïve café", ascii=True), "naive_cafe")

    def test_handle_numerals_as_is(self):
        self.assertEqual(clean_names("Chapter 10", numerals="asis"), "chapter_10")

    def test_handle_numerals_remove(self):
        self.assertEqual(clean_names("Chapter 10", numerals="remove"), "chapter")

    def test_handle_numerals_spell(self):
        self.assertEqual(clean_names("Chapter 10", numerals="spell"), "chapter_ten")

    def test_duplicate_names_removal(self):
        self.assertEqual(clean_names("test.test.test", allow_dupes=False), "test.test_2.test_3")

# Run the tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestCleanNamesFunction))