# helper_py
This repo contains functions designed to provide assistance in working with data.


## clean_names.py

This script offers utilities to clean and transform string names. Here are its core features:

**Main Function**: 
- `clean_names()`: Accepts an input string and a set of parameters to clean and transform the string based on user preferences.

The script also contains several helper functions to provide the user with multiple features.

1. **Transliteration Rules**: Provides built-in rules for Greek characters and offers support for Latin, ASCII, and Cyrillic transliterations.
2. **Case Styles**: Supports multiple case styles including snake, camel, pascal, kebab, and more.
3. **Numerals Handling**: Can remove, keep as-is, or spell out numeric values in strings.
4. **String Sanitization**: Removes unwanted characters, separators, and can ensure the cleaned string is a valid variable name.
5. **Custom Character Replacement**: Allows for custom character replacement based on user-defined rules.
