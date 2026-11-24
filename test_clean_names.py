import importlib.util
import logging
import time
import unicodedata

import pytest

from clean_names import clean_names, clean_names_table, remove_duplicate_names

HAS_NUM2WORDS = importlib.util.find_spec("num2words") is not None
needs_num2words = pytest.mark.skipif(not HAS_NUM2WORDS, reason="num2words not installed")


# ---------------- Original UAT (must stay green) ----------------

@pytest.mark.parametrize("case, text, expected", [
    ("snake", "This is a Test", "this_is_a_test"),
    ("camel", "This is a Test", "thisIsATest"),
    ("pascal", "This is a Test", "ThisIsATest"),
    ("kebab", "This is a Test", "this-is-a-test"),
    ("space", "This is a Test", "this is a test"),
    ("title", "this is a test", "This_Is_A_Test"),
    ("upper", "this is a test", "THIS_IS_A_TEST"),
    ("lower", "This IS a TEST", "this_is_a_test"),
    ("sentence", "this is a test", "This_is_a_test"),
    ("snake_case", "This is a Test", "this_is_a_test"),
])
def test_case_styles(case, text, expected):
    assert clean_names(text, case=case) == expected


@pytest.mark.parametrize("text, kwargs, expected", [
    ("100% of #1", {}, "100_percent_of_number_1"),
    ("100% of #1", {"replace": {"%": "percent", "#": "number"}}, "100percent_of_number1"),
    ("Chapter 10", {"numerals": "asis"}, "chapter_10"),
    ("Chapter 10", {"numerals": "keep"}, "chapter_10"),
    ("Chapter 10", {"numerals": "remove"}, "chapter"),
    ("test.test.test", {}, "test.test_2.test_3"),
    ("A.B.% #", {"case": "old_janitor"}, "a_b_percent_number"),
    ("καλημέρα", {"transliterations": "Greek"}, "kalhmera"),
    ("καλημέρα", {"transliterations": "Greek", "ascii": False, "transliterate": True}, "kalhmera"),
    ("test..test", {"sep_in": "."}, "test.test_2"),
    ("a/b.c", {"sep_in": ["/", "."]}, "a.b.c"),
    ("a::b::c", {"sep_in": ["::"]}, "a.b.c"),
    ("a/b/c", {"sep_in": "/", "sep_out": "__"}, "a__b__c"),
    ("abc123def", {"replace": {r"\d+": "_NUM_"}, "replace_regex": True}, "abc_num_def"),
    ("naïve café", {}, "naive_cafe"),
    ("123x", {}, "123x"),
])
def test_original_behaviour(text, kwargs, expected):
    assert clean_names(text, **kwargs) == expected


@needs_num2words
def test_numerals_spell():
    assert clean_names("Chapter 10", numerals="spell") == "chapter_ten"


def test_digits_only_without_num2words_raises():
    if HAS_NUM2WORDS:
        pytest.skip("num2words installed")
    with pytest.raises(ModuleNotFoundError, match="pip install num2words"):
        clean_names("123")


def test_list_input_raises():
    with pytest.raises(ValueError):
        clean_names(["a", "b"])


def test_table():
    assert clean_names_table(["This is a Test", "Chapter 10", "test.test.test"]) == [
        ("This is a Test", "this_is_a_test"),
        ("Chapter 10", "chapter_10"),
        ("test.test.test", "test.test_2.test_3"),
    ]
    assert clean_names_table([None, "x"], skip_none=True) == [("x", "x")]
    assert clean_names_table([None, "123x", "naïve café"]) == [("", ""), ("123x", "123x"), ("naïve café", "naive_cafe")]


@needs_num2words
def test_table_int_is_spelled():
    (orig, new), = clean_names_table([123])
    assert orig == "123" and new == "one_hundred_and_twenty_three"


def test_performance_smoke():
    t0 = time.perf_counter()
    for _ in range(5000):
        clean_names("This is a Test 123")
    assert time.perf_counter() - t0 < 2.0


# ---------------- Regressions from the adversarial review ----------------

@pytest.mark.parametrize("text, expected", [
    ("Sales ($)", "sales"),                      # punctuation used to leak: "sales_($)"
    ("a-b (c)/d & e?", "a_b_c_d_e"),
    ("Revenue – Q1", "revenue_q1"),              # en dash is a separator, not glue
    ("  leading and trailing  ", "leading_and_trailing"),
    ("test.", "test"),                           # trailing separator used to survive
    ("..a..b..", "a.b"),
    ("a.$.b", "a.b"),                            # segment that cleans to nothing is dropped
    ("", ""),
    ("!!!", ""),
])
def test_junk_is_removed(text, expected):
    assert clean_names(text) == expected


@pytest.mark.parametrize("case, expected", [
    ("snake", "hello_world"), ("kebab", "hello-world"), ("upper", "HELLO_WORLD"),
    ("title", "Hello_World"), ("space", "hello world"), ("camel", "helloWorld"),
])
def test_camel_input_split_consistently_across_styles(case, expected):
    assert clean_names("HelloWorld", case=case) == expected


def test_acronyms_and_digits():
    assert clean_names("HTTPServer") == "http_server"
    assert clean_names("COVID19Cases") == "covid19_cases"
    assert clean_names("version_2b", case="camel") == "version2b"   # str.title() gave "version2B"
    assert clean_names("x2y", case="title") == "X2y"


@pytest.mark.parametrize("text, expected", [
    ("Straße", "strasse"), ("Łódź", "lodz"), ("Ærø", "aero"), ("ÆRØ", "aero"),
    ("Привет мир", "privet_mir"),               # was "" with default options
    ("Щука Йошкар-Ола", "shchuka_yoshkar_ola"),
    ("καλημέρα κόσμε", "kalhmera_kosme"),       # was ""
    ("ΘΕΑ", "thea"), ("Θέα", "thea"),            # digraph case does not break word splitting
    ("ﬁle①", "file1"),                          # NFKD compatibility folding
    ("5 µg/kg", "5_ug_kg"),                      # micro sign -> u, not Greek mu -> m
])
def test_transliteration(text, expected):
    assert clean_names(text) == expected


def test_ascii_true_guarantees_ascii():
    out = clean_names("中文 Ελληνικά Русский ☃ naïve", warn_unhandled_unicode=False)
    assert out.isascii()


def test_dropped_letters_warn(caplog):
    import clean_names as module
    if module.unidecode is not None:
        pytest.skip("unidecode rescues these characters")
    with caplog.at_level(logging.WARNING, logger="clean_names"):
        assert clean_names("a中b") == "a_b"             # dropped text still separates words
    assert "中" in caplog.text


def test_non_ascii_mode_keeps_words_intact():
    assert clean_names("Größe Ärger", ascii=False) == "größe_ärger"
    assert clean_names("हिन्दी नाम", ascii=False) == "हिन्दी_नाम"   # combining marks stay inside words
    nfd = unicodedata.normalize("NFD", "naïve")
    assert clean_names(nfd, ascii=False) == unicodedata.normalize("NFC", "naïve")  # NFD input is normalized
    assert clean_names("²", ascii=False) == "²"                  # used to crash in int("²")


def test_latin_step_only_touches_latin():
    assert clean_names("café й", ascii=False, transliterate=True, transliterations="latin") == "cafe_й"


def test_old_janitor_does_not_dedupe_words():
    assert clean_names("Bora Bora", case="old_janitor") == "bora_bora"   # was "bora_bora_2"
    assert clean_names("HelloWorld", case="old_janitor") == "helloworld"


def test_dedupe_is_per_segment_regardless_of_sep_out():
    assert clean_names("a.a b", sep_out="_") == "a_a_b"      # was "a_a_2_b"
    assert clean_names("x.x", sep_out="_") == "x_x_2"


def test_remove_duplicate_names_never_collides():
    assert remove_duplicate_names(["a", "a", "a_2"]) == ["a", "a_3", "a_2"]
    assert remove_duplicate_names(["a", "a_2", "a", "a"]) == ["a", "a_2", "a_3", "a_4"]
    out = remove_duplicate_names(["x"] * 5 + ["x_3"])
    assert len(set(out)) == len(out)


def test_replacements():
    assert clean_names("x", replace={"": "Z"}) == "x"                                  # was "zx_z"
    assert clean_names("50%%", replace={"%": "pct", "%%": "permille"}) == "50permille"  # longest wins
    assert clean_names("a.b", sep_out=r"\1") == r"a\1b"                               # was re.error


@pytest.mark.parametrize("sep_in, text, expected", [
    ("", "a.b", "a_b"),          # disables splitting (was re.error)
    ([], "a.b", "a_b"),
    ("\\", "a\\b", "a.b"),       # lone backslash is literal (was re.error)
    (r"\.\-", "a.b-c", "a.b.c"), # legacy char-class fragment still works
    ({"/"}, "a/b", "a.b"),
])
def test_sep_in_forms(sep_in, text, expected):
    assert clean_names(text, sep_in=sep_in) == expected


@pytest.mark.parametrize("kwargs", [
    {"case": "snek"}, {"numerals": "bogus"}, {"parsing_option": "two"},
    {"transliterations": "greeek"}, {"sep_out": None},
])
def test_bad_options_fail_fast(kwargs):
    with pytest.raises((ValueError, TypeError)):
        clean_names("x", **kwargs)


def test_bad_options_fail_before_any_row():
    with pytest.raises(ValueError):
        clean_names_table(iter(["a"]), case="nope")


@pytest.mark.parametrize("bad", [("a", "b"), {"a"}, {"a": 1}])
def test_collections_rejected(bad):
    with pytest.raises(ValueError):
        clean_names(bad)


def test_table_rejects_bare_string():
    with pytest.raises(ValueError):
        clean_names_table("Hello")       # used to emit one row per character


def test_table_unique():
    rows = clean_names_table(["a b", "A-B", "a_b"], unique=True)
    assert [r[1] for r in rows] == ["a_b", "a_b_2", "a_b_3"]


def test_case_aliases():
    for alias in ("kebab-case", "KebabCase", "kc", "kebab case"):
        assert clean_names("Hello World", case=alias) == "hello-world"
    assert clean_names("Hello World", case="SCREAMING_SNAKE") == "HELLO_WORLD"


@needs_num2words
def test_spell_inserts_word_boundaries():
    assert clean_names("Chapter21", numerals="spell") == "chapter_twenty_one"   # no stray hyphen
    assert clean_names(" 123 ") == "one_hundred_and_twenty_three"


def test_library_does_not_print(capsys):
    clean_names("anything at all")
    assert capsys.readouterr() == ("", "")
