# AppropriatenessEvaluator.py

# --------------------- #
#      Setup & Install  #
# --------------------- #

# Install necessary libraries
!pip install openai sentence-transformers scikit-learn nltk pandas scipy

# --------------------- #
#      Import Libraries #
# --------------------- #

import os
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import openai
import numpy as np
from getpass import getpass
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# --------------------- #
#      Configure Logging #
# --------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --------------------- #
#    Download NLTK Data  #
# --------------------- #

nltk.download('stopwords')
nltk.download('wordnet')

# --------------------- #
#   Preprocessing Function #
# --------------------- #

def preprocess_item(item: str) -> str:
    """
    Preprocesses a single item by lowercasing, removing special characters,
    tokenizing, removing stopwords, and lemmatizing.

    Args:
        item (str): The item string to preprocess.

    Returns:
        str: The preprocessed item.
    """
    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase
    item = item.lower()
    # Remove special characters
    item = ''.join(e for e in item if e.isalnum() or e.isspace())
    # Tokenize
    tokens = item.split()
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Rejoin tokens
    return ' '.join(tokens)

# --------------------- #
#    Create Sample Dataset #
# --------------------- #

def create_sample_dataset(context: str) -> pd.DataFrame:
    """
    Creates a sample dataset based on the provided context.

    Args:
        context (str): The context description.

    Returns:
        pd.DataFrame: DataFrame containing item IDs and their descriptions.
    """
    # Placeholder dataset; in practice, replace this with actual data
    data = {
        'item_id': [1, 2, 3, 4, 5],
        'items': [
            'spaghetti, garlic, tomato sauce, basil, olive oil, parmesan',
            'penne, garlic, tomato, onion, basil, olive oil, feta cheese',
            'fettuccine, cream, garlic, parmesan, black pepper, parsley',
            'linguine, garlic, shrimp, lemon, butter, parsley',
            'rigatoni, sausage, tomato sauce, onion, bell pepper, parmesan'
        ]
    }
    dataset = pd.DataFrame(data)
    return dataset

# --------------------- #
#     Load & Preprocess Data #
# --------------------- #

def load_and_preprocess_data(dataset: pd.DataFrame, item_column: str = 'items') -> pd.DataFrame:
    """
    Loads the dataset and preprocesses the items.

    Args:
        dataset (pd.DataFrame): DataFrame containing items.
        item_column (str): The column name containing items.

    Returns:
        pd.DataFrame: DataFrame with preprocessed items.
    """
    logging.info("Preprocessing items...")
    # Apply preprocessing to each item list
    dataset['processed_items'] = dataset[item_column].apply(
        lambda x: ', '.join([preprocess_item(i) for i in x.split(',')])
    )
    return dataset

# --------------------- #
#      Label Items  #
# --------------------- #

def label_items(dataset: pd.DataFrame, frequency_threshold: float = 0.4, item_column: str = 'processed_items') -> pd.DataFrame:
    """
    Labels items as appropriate (1) or inappropriate (0) based on their frequency.

    Args:
        dataset (pd.DataFrame): DataFrame containing preprocessed items.
        frequency_threshold (float): Minimum frequency (as a proportion) to label as appropriate.
        item_column (str): The column name containing processed items.

    Returns:
        pd.DataFrame: DataFrame with an additional 'label' column.
    """
    logging.info("Labeling items based on frequency...")
    # Explode the items into individual rows
    all_items = dataset[item_column].str.split(', ').explode()
    # Calculate item frequencies
    item_counts = all_items.value_counts()
    total_entries = len(dataset)
    # Define label function
    def label_item(item: str) -> int:
        freq = item_counts.get(item, 0) / total_entries
        return 1 if freq >= frequency_threshold else 0  # 1: appropriate, 0: inappropriate
    
    # Apply labeling to each item in each entry
    # Here, we label the entire entry as inappropriate if any item is inappropriate
    dataset['item_labels'] = dataset[item_column].apply(
        lambda x: [label_item(i) for i in x.split(', ')]
    )
    # Define entry label
    dataset['label'] = dataset['item_labels'].apply(lambda labels: 0 if 0 in labels else 1)
    return dataset

# --------------------- #
#    Feature Extraction  #
# --------------------- #

def vectorize_features(dataset: pd.DataFrame, item_column: str = 'processed_items') -> Tuple[csr_matrix, TfidfVectorizer, SentenceTransformer]:
    """
    Vectorizes the preprocessed items using TF-IDF and word embeddings.

    Args:
        dataset (pd.DataFrame): DataFrame containing preprocessed items.
        item_column (str): The column name containing processed items.

    Returns:
        Tuple[csr_matrix, TfidfVectorizer, SentenceTransformer]:
            - Combined feature matrix.
            - Fitted TF-IDF vectorizer.
            - Loaded SentenceTransformer model.
    """
    logging.info("Vectorizing features using TF-IDF...")
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    # Fit and transform the processed items
    tfidf_matrix = tfidf.fit_transform(dataset[item_column])
    
    logging.info("Generating embeddings using SentenceTransformer...")
    # Initialize SentenceTransformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings
    embeddings = model.encode(dataset[item_column].tolist(), convert_to_tensor=False)
    
    logging.info("Combining TF-IDF vectors with embeddings...")
    # Convert embeddings to sparse matrix
    embeddings_sparse = csr_matrix(embeddings)
    # Combine features
    combined_features = hstack([tfidf_matrix, embeddings_sparse])
    
    return combined_features, tfidf, model

# --------------------- #
#     Train Classifier   #
# --------------------- #

def train_classifier(X: csr_matrix, y: pd.Series) -> Ridge:
    """
    Trains a Ridge Regression model to predict appropriateness scores.

    Args:
        X (csr_matrix): Feature matrix.
        y (pd.Series): Labels (1: Appropriate, 0: Inappropriate).

    Returns:
        Ridge: Trained regression model.
    """
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logging.info("Training Ridge Regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    return model

# --------------------- #
#   Conditional Probability  #
# --------------------- #

def get_conditional_probability(item: str, context: str, model_name: str = "text-davinci-003") -> float:
    """
    Retrieves the conditional probability of an item given a context from the LLM.

    Args:
        item (str): The item to get the probability for.
        context (str): The context/prompt to provide to the LLM.
        model_name (str): The LLM model to use.

    Returns:
        float: The conditional probability of the item. Returns 0.0 if not found.
    """
    try:
        prompt = context + item
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=0,  # We don't need actual tokens
            logprobs=100,  # Increase to cover more tokens
            echo=True,
            temperature=0  # Deterministic output
        )
        # Extract log probabilities
        # Since max_tokens=0, we need to adjust to get the logprob of the last token
        # This approach may vary depending on OpenAI API capabilities
        # Placeholder implementation:
        log_probs = response['choices'][0]['logprobs']['token_logprobs']
        tokens = response['choices'][0]['logprobs']['tokens']
        if not tokens:
            return 0.0
        # Get the last token's log probability
        last_token = tokens[-1].strip().lower()
        item_lower = item.lower()
        if last_token == item_lower:
            return np.exp(log_probs[-1])  # Convert log prob to probability
        else:
            return 0.0
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error for item '{item}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error retrieving probability for '{item}': {e}")
    return 0.0

# --------------------- #
#   Density-Based Outlier Detection  #
# --------------------- #

def fit_density_model(features: csr_matrix) -> LocalOutlierFactor:
    """
    Fits a LocalOutlierFactor model on the feature set.

    Args:
        features (csr_matrix): Feature matrix.

    Returns:
        LocalOutlierFactor: Fitted LOF model.
    """
    logging.info("Fitting Local Outlier Factor model on appropriate items...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(features)
    return lof

def is_outlier(feature: csr_matrix, lof_model: LocalOutlierFactor) -> bool:
    """
    Determines if a feature vector is an outlier.

    Args:
        feature (csr_matrix): Feature vector.
        lof_model (LocalOutlierFactor): Fitted LOF model.

    Returns:
        bool: True if outlier, False otherwise.
    """
    return lof_model.predict(feature) == -1

# --------------------- #
#   Evaluate New Item #
# --------------------- #

def evaluate_new_item(
    new_item: str,
    tfidf: TfidfVectorizer,
    embedding_model: SentenceTransformer,
    classifier: Ridge,
    context: str,
    nn_model: NearestNeighbors,
    lof_model: LocalOutlierFactor,
    scaler: StandardScaler,
    embedding_matrix: np.ndarray,
    similarity_threshold: float = 0.7,
    score_threshold: float = 0.5
) -> str:
    """
    Evaluates whether a new item is appropriate within the given context.

    Args:
        new_item (str): The item to evaluate.
        tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer.
        embedding_model (SentenceTransformer): Loaded SentenceTransformer model.
        classifier (Ridge): Trained regression model.
        context (str): Context/prompt for LLM.
        nn_model (NearestNeighbors): Trained nearest neighbors model for similarity.
        lof_model (LocalOutlierFactor): Trained LOF model for outlier detection.
        scaler (StandardScaler): Scaler fitted on training data.
        embedding_matrix (np.ndarray): Matrix of embeddings for similarity calculations.
        similarity_threshold (float): Threshold for cosine similarity.
        score_threshold (float): Threshold for classifier decision.

    Returns:
        str: "Appropriate" or "Inappropriate"
    """
    logging.info(f"Evaluating new item: '{new_item}'")
    # Preprocess
    processed = preprocess_item(new_item)
    
    # TF-IDF Vectorization
    tfidf_vector = tfidf.transform([processed])
    
    # Embedding
    embedding = embedding_model.encode([processed], convert_to_tensor=False)
    embedding_sparse = csr_matrix(embedding)
    
    # Combine Features
    combined_feature = hstack([tfidf_vector, embedding_sparse])
    
    # Normalize if scaler is provided
    if scaler:
        combined_feature = scaler.transform(combined_feature)
    
    # LLM Conditional Probability
    llm_probability = get_conditional_probability(new_item, context)
    # Convert probability to log probability
    llm_log_prob = np.log(llm_probability + 1e-10)  # Add epsilon to avoid log(0)
    llm_log_prob_feature = csr_matrix([llm_log_prob])
    
    # Final Combined Feature (Appending LLM log probability)
    final_feature = hstack([combined_feature, llm_log_prob_feature])
    
    # Predict using classifier
    predicted_score = classifier.predict(final_feature)[0]
    
    # Initial Decision based on classifier score
    decision = "Appropriate" if predicted_score >= score_threshold else "Inappropriate"
    
    # Similarity Adjustment
    ingredient_embedding = embedding
    similarities = cosine_similarity(ingredient_embedding, embedding_matrix)
    max_similarity = np.max(similarities)
    logging.info(f"Maximum cosine similarity: {max_similarity:.4f}")
    
    if max_similarity < similarity_threshold and predicted_score >= score_threshold:
        decision = "Inappropriate"
        logging.info("Adjusted decision based on low similarity to known appropriate items.")
    
    # Density-Based Outlier Detection
    if is_outlier(final_feature, lof_model):
        decision = "Inappropriate"
        logging.info("Adjusted decision based on density-based outlier detection.")
    
    logging.info(f"Final Decision for '{new_item}': {decision}")
    return decision

# --------------------- #
#        Main Function   #
# --------------------- #

def main():
    """
    Main function to execute the automated appropriateness evaluation system.
    """
    # Configuration
    CONTEXT = "List common ingredients in a pasta recipe: garlic, parmesan, tomato, onion, basil, olive oil, "
    TARGET_ITEMS = ["Cheetos", "chocolate", "tofu", "anchovies", "pineapple", 
                    "beef jerky", "kale", "quinoa", "seitan", "almonds"]  # Example target items
    
    # Step 1: Create and Preprocess Data
    dataset = create_sample_dataset(context=CONTEXT)
    dataset = load_and_preprocess_data(dataset, item_column='items')
    
    # Step 2: Label Items
    dataset = label_items(dataset, frequency_threshold=0.4, item_column='processed_items')  # Adjust threshold as needed
    
    # Step 3: Feature Extraction
    combined_features, tfidf_vectorizer, embedding_model = vectorize_features(dataset, item_column='processed_items')
    
    # Step 4: Train Classifier
    classifier = train_classifier(combined_features, dataset['label'])
    
    # Step 5: Fit Nearest Neighbors on Appropriate Items
    logging.info("Fitting Nearest Neighbors model on appropriate items...")
    # Extract appropriate items
    appropriate_indices = dataset[dataset['label'] == 1].index
    appropriate_features = combined_features[appropriate_indices]
    
    # Initialize and fit NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(appropriate_features)
    
    # Step 6: Fit Density Model
    lof_model = fit_density_model(appropriate_features)
    
    # Step 7: Fit Scaler on Combined Features
    logging.info("Fitting StandardScaler on combined features...")
    scaler = StandardScaler(with_mean=False)
    scaler.fit(combined_features)
    
    # Step 8: Prepare Embedding Matrix for Similarity Calculations
    logging.info("Preparing embedding matrix for similarity calculations...")
    embedding_matrix = embedding_model.encode(dataset['processed_items'].tolist(), convert_to_tensor=False)
    
    # Step 9: Evaluate New Items
    print("\n=== Item Evaluation Results ===")
    for item in TARGET_ITEMS:
        decision = evaluate_new_item(
            new_item=item,
            tfidf=tfidf_vectorizer,
            embedding_model=embedding_model,
            classifier=classifier,
            context=CONTEXT,
            nn_model=nn_model,
            lof_model=lof_model,
            scaler=scaler,
            embedding_matrix=embedding_matrix,
            similarity_threshold=0.7,  # Adjust as needed
            score_threshold=0.5       # Adjust as needed
        )
        print(f"{item}: {decision}")

# --------------------- #
#       Execute Main     #
# --------------------- #

if __name__ == "__main__":
    # Prompt user for OpenAI API key securely
    api_key = getpass("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    main()
