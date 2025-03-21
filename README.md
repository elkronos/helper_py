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

## appropriateness_evaluator.py

This script provides a comprehensive utility for evaluating whether given items (e.g., ingredients) are “appropriate” within a specified context. The core of the utility is the `AppropriatenessEvaluator` class, which combines traditional NLP techniques with modern machine learning and LLM-based methods to assess item suitability.

**Main Class**:
- `AppropriatenessEvaluator`:
  - Initializes with a context description, an optional dataset (or uses a sample dataset), and various configurable thresholds.
  - Preprocesses text by normalizing, tokenizing, removing stopwords, and lemmatizing each item.
  - Labels dataset entries as appropriate/inappropriate based on item frequency.
  - Constructs a unified feature matrix by combining TF–IDF vectors and SentenceTransformer embeddings.
  - Trains a Ridge regression classifier to predict item appropriateness.
  - Enhances evaluation using nearest neighbors for similarity checking, Local Outlier Factor (LOF) for density-based outlier detection, and LLM-based conditional probability checks via the OpenAI API.
  - Provides methods to evaluate both individual and multiple items interactively.

### Key Features:

1. **Advanced Text Preprocessing**:
   - Converts text to lowercase and removes special characters.
   - Tokenizes, removes stopwords, and lemmatizes tokens to standardize input data.

2. **Feature Engineering & Model Training**:
   - Utilizes TF–IDF vectorization alongside SentenceTransformer embeddings.
   - Combines sparse and dense representations into a single feature matrix.
   - Trains a Ridge regression classifier to score item appropriateness.

3. **Similarity and Density-Based Adjustments**:
   - Applies nearest neighbors analysis to measure cosine similarity with known appropriate items.
   - Uses Local Outlier Factor (LOF) to detect density-based anomalies among items.

4. **LLM-Enhanced Evaluation**:
   - Retrieves conditional probability scores from an LLM (via OpenAI API) to further refine decisions.
   - Overrides classifier decisions if the LLM probability falls below a specified threshold.

5. **Interactive and Flexible Data Input**:
   - Supports custom datasets via CSV or a built-in sample dataset.
   - Provides an interactive CLI to input context and items for evaluation, along with secure API key handling.

6. **Robust Logging and Error Handling**:
   - Implements detailed logging for tracking processing steps and debugging.
   - Includes error management for API calls, model predictions, and data processing issues.
  

## audio_directory_scanner.py

This Python script provides a utility for scanning directories, collecting metadata about files, and saving the gathered information into a CSV file. It is implemented using the Tkinter library for the GUI, allowing users to select a directory and view a progress bar during the scanning process. The script is particularly useful for cataloging files, including media files, by recording their duration.

**Main Class**:
- `DirectoryScannerApp`: The primary class that encapsulates the directory scanning functionality, file metadata collection, and user interaction through a Tkinter-based GUI.

### Key Features:

1. **Directory Scanning**:
   - Recursively scans the selected directory and its subdirectories.
   - Captures the file structure, including folder hierarchy and file types.
   - Extracts the duration of media files (e.g., MP3, WAV, FLAC) using the Mutagen library.

2. **CSV Export**:
   - Saves the scanned file data to a CSV file, with columns dynamically adjusted based on directory depth.
   - Automatically generates a timestamped CSV filename based on the directory name.
   - Includes the file duration in the output for supported media files.

3. **User Interface**:
   - Utilizes Tkinter for a simple and interactive GUI.
   - Provides dialog boxes for directory selection, error handling, and success notifications.
   - Includes a progress bar that updates during the scanning process, keeping the user informed.

4. **Error Handling**:
   - Robust error handling with user-friendly messages for various issues like directory access errors or file processing failures.
   - Ensures graceful application exit even in the event of errors.


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


## deduplicate_dataframe.py

This script provides utilities to deduplicate pandas DataFrames. Here are its core features:

**Main Function**: 
- `deduplicate_dataframe()`: Accepts a DataFrame and a set of parameters to remove or flag duplicates based on user preferences.

The script also contains several helper functions to provide the user with multiple features.

1. **Column-Specific Deduplication**: Allows specifying one or more columns to check for duplicates.
2. **Flagging Duplicates**: Option to flag duplicates instead of removing them by adding a boolean column.
3. **Tie-Breaking Criteria**: Supports keeping the first or last occurrence of duplicates, or using a custom function.
4. **Grouping**: Ability to group by specified columns before checking for duplicates.
5. **Parameter Validation**: Ensures all provided parameters are valid and handles various edge cases.


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


## extract_archives.py

This script automates the extraction of ZIP and RAR archives within a specified directory. It supports recursive extraction of nested archives, parallel processing for efficiency, and robust error handling. Additionally, the script provides options for verbosity, automatic deletion of archives after successful extraction, and simulation of the extraction process without making any changes.

**Main Function**:
- `process_directory()`: Traverses the given directory, identifies ZIP and RAR files, and extracts them in parallel batches. It handles nested archives iteratively, manages retries for failed extractions, and cleans up empty directories post-extraction.

### Key Features:

1. **Comprehensive Archive Support**:
   - Extracts both ZIP and RAR file formats.
   - Validates archive integrity before extraction to ensure data consistency.

2. **Parallel Processing**:
   - Utilizes `ThreadPoolExecutor` for concurrent extraction, significantly speeding up the processing of multiple archives.
   - Configurable number of worker threads to optimize performance based on system capabilities.

3. **Nested Archive Handling**:
   - Automatically detects and extracts archives contained within other archives, ensuring thorough processing of all nested files.

4. **Robust Logging**:
   - Logs detailed information about the extraction process, including successes, failures, and actions taken.
   - Outputs logs to both the console and a dedicated log file (`extract_archives.log`) for easy monitoring and troubleshooting.

5. **Flexible Deletion Options**:
   - Option to automatically delete archives after successful extraction.
   - Interactive prompts for manual approval of archive deletion, with an option to approve all subsequent deletions.

6. **Error Handling and Recovery**:
   - Moves problematic archives to a `failed_archives` directory for later review.
   - Implements retry mechanisms for transient extraction failures, enhancing reliability.

7. **Dry-Run Capability**:
   - Simulate the extraction process without making any actual changes, allowing users to verify actions before execution.

8. **Customizable Exclusions**:
   - Excludes specified files and directories from processing based on predefined patterns or user-defined criteria.
   - Easily extendable to incorporate more complex exclusion logic as needed.

9. **User-Friendly Command-Line Interface**:
   - Intuitive argument parsing for specifying directories, verbosity, deletion preferences, and more.
   - Provides clear feedback and progress indicators during execution, including a dynamic progress bar.

### Usage:

```bash
python extract_archives.py /path/to/directory [options]
```

**Arguments**:
- `directory` (str): Directory to process.

**Options**:
- `--verbose`: Increase output verbosity.
- `--auto-delete`: Automatically approve all deletions.
- `--unrar-path`: Specify the path to `UnRAR.exe` if different from the default.
- `--dry-run`: Simulate the extraction process without making changes.
- `--max-workers` (int): Maximum number of worker threads for parallel extraction (default: 4).
- `--batch-size` (int): Number of archives to process per batch (default: 100).


## ProjectPath.py

This module introduces the `ProjectPath` class, a robust solution for navigating and managing paths within Python projects. Here's a look at what it offers:

**Core Class**:
- `ProjectPath`: Facilitates dynamic identification and handling of paths relative to the project's root directory, based on customizable markers.

**Features**:
1. **Dynamic Project Root Discovery**: Utilizes directory markers (e.g., `.git`, `.svn`) to automatically find the project root.
2. **Adaptable to Environment**: Capable of determining the start path correctly, whether executed in a script or interactive environments such as Jupyter notebooks.
3. **Marker Customization**: Markers can be dynamically added, allowing for flexible adjustment to root discovery criteria.
4. **Efficient Root Discovery**: Incorporates a caching mechanism to optimize the process of finding the project root across multiple operations.
5. **Convenient Path Manipulation**: Provides a straightforward way to construct paths from the project root to various resources within the project.
6. **Clear Error Reporting**: Offers informative error messages for cases where the project root cannot be determined with the given markers.

**Example Usage**:
```python
# Initialize with specific markers and path parts
project_file = ProjectPath("data", "mydata.csv", markers=['.git', 'my_project_marker.file'])
print(project_file)  # Outputs the path as a string
print(project_file.path())  # Returns the path as a Path object
