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


## detect_sentiments_and_emotions.py

This script provides comprehensive functionality for sentiment and emotion analysis in text data. It offers both word-level and sentence-level sentiment analysis using different methods.

### Main Functions:

- `analyze_sentiment(text, level='sentence')`: Analyzes the sentiment of a given text. It supports two levels of analysis:
  - Sentence-level: Utilizes a pre-trained BERT model for nuanced sentiment analysis of the entire text.
  - Word-level: Uses the Afinn library to provide sentiment scores for individual words.

- `process_text(text, sentiment_level='sentence')`: Processes a given text to extract words, perform sentiment analysis at the specified level (sentence or word), and identifies emotions based on a lexicon.

- `detect_sentiments_and_emotions(df, columns, sentiment_level='sentence')`: Processes a DataFrame to detect sentiments and emotions in specified columns. It applies either sentence-level or word-level sentiment analysis to each text and aggregates the results.


### Key Components:

- **AFIN Word-level Sentiment Analysis**: Utilizes the Afinn library, which assigns sentiment scores to individual words based on a predefined lexicon.
- **BERT Sentence-level Sentiment Analysis**: Leverages the `nlptown/bert-base-multilingual-uncased-sentiment` model from Hugging Face's transformers library, offering robust and context-aware sentiment analysis for complete sentences.
- **Emotion Lexicon**: Includes a basic emotion lexicon mapping words to emotions like joy, sadness, surprise, and anger. This can be expanded as needed.
- **Pandas DataFrame Integration**: Designed to work with pandas DataFrames, making it suitable for analyzing tabular data containing text.
- **String Manipulation**: Employs Python's `string` module for basic text cleaning, like stripping punctuation and converting to lower case.
