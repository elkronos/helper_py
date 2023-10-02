# helper_py
This repo contains functions designed to provide assistance in working with data.


## apply_weights.py

This script provides a utility for applying demographic weights to numerical columns in a Pandas DataFrame. The main function, `apply_weights()`, accommodates multiple demographic variables and corresponding weights, creating new weighted columns in the DataFrame. This utility is particularly useful in survey data analysis, ensuring that the analyzed data is representative of the actual population.

**Main Function**:
- `apply_weights()`: Accepts a DataFrame, demographic variables, and demographic weights, applying these weights to specific numeric columns and creating new weighted columns in the DataFrame.

### Key Features:

1. **Demographic Weighting**:
   - Apply weights to numeric columns based on demographic categorizations.
   - Handle multiple demographic variables and associated weights.
2. **Dynamic Column Creation**:
   - Generate new columns with weighted values, preserving original numeric columns.
   - Specify new column names or auto-generate them based on original names and user-defined suffixes.
3. **Flexible Weight Application**:
   - Apply weights to all numeric columns or a user-specified subset.
   - Optionally display intermediate weightings for each demographic variable (verbose mode).
4. **Data Integrity**:
   - Conduct input checks for type and value validation, ensuring consistency in the weighting process.
   - Manage missing values to maintain accurate weight application.


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
