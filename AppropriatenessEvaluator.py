#!/usr/bin/env python
"""
AppropriatenessEvaluator Module

This module implements a system to evaluate whether an item is “appropriate”
within a given context. It preprocesses text data, creates TF-IDF and embedding
features, scales the features, trains a Ridge regression model, and uses additional
signals (LLM conditional probability, cosine similarity, and density‐based outlier
detection) to make a final decision.
"""

import os
import logging
import re
from getpass import getpass
from typing import List, Tuple, Optional

import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import openai
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity

# --------------------- #
#   Configure Logging   #
# --------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# --------------------- #
#   Download NLTK Data  #
# --------------------- #

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# --------------------- #
#  Preprocessing Code   #
# --------------------- #

def preprocess_item(item: str) -> str:
    """
    Preprocesses a string by lowercasing, removing non-alphanumeric characters,
    removing stopwords, and lemmatizing.

    Args:
        item (str): The input text.

    Returns:
        str: The processed text.
    """
    if not isinstance(item, str):
        return ""
    # Lowercase and remove non-alphanumeric characters (keep spaces)
    item = item.lower()
    item = re.sub(r'[^a-z0-9\s]', '', item)
    tokens = item.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(processed_tokens)


# ------------------------------ #
#   AppropriatenessEvaluator     #
# ------------------------------ #

class AppropriatenessEvaluator:
    def __init__(
        self,
        context: str,
        dataset: Optional[pd.DataFrame] = None,
        item_column: str = 'items',
        frequency_threshold: float = 0.4,
        similarity_threshold: float = 0.7,
        score_threshold: float = 0.5,
        llm_probability_threshold: float = 0.01,
        model_name: str = "text-davinci-003",
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initializes the evaluator with context, dataset, and various thresholds.

        Args:
            context (str): Description of the context.
            dataset (Optional[pd.DataFrame]): DataFrame with items (if None, a sample dataset is used).
            item_column (str): Column name in the dataset that contains the items.
            frequency_threshold (float): Minimum frequency for an item to be considered appropriate.
            similarity_threshold (float): Minimum cosine similarity for similarity-based adjustment.
            score_threshold (float): Classifier score threshold for deciding appropriateness.
            llm_probability_threshold (float): Minimum LLM conditional probability to pass.
            model_name (str): OpenAI model name.
            embedding_model_name (str): SentenceTransformer embedding model name.
        """
        self.context = context
        self.dataset = dataset if dataset is not None else self.create_sample_dataset()
        self.item_column = item_column
        self.frequency_threshold = frequency_threshold
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.llm_probability_threshold = llm_probability_threshold
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        # Pipeline: Preprocessing, labeling, vectorization, scaling, and training
        self.preprocess_data()
        self.label_items()
        self.vectorize_features()
        self.fit_scaler_and_transform()  # Scale features before model training
        self.train_classifier()
        self.fit_nearest_neighbors()
        self.fit_density_model()
        self.prepare_embedding_matrix()

    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Creates a sample dataset if none is provided.

        Returns:
            pd.DataFrame: A sample dataset with item IDs and descriptions.
        """
        logging.info("Creating a sample dataset...")
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
        return pd.DataFrame(data)

    def preprocess_data(self) -> None:
        """
        Applies text preprocessing to the dataset items.
        """
        logging.info("Preprocessing dataset items...")
        self.dataset['processed_items'] = self.dataset[self.item_column].apply(
            lambda x: ', '.join([preprocess_item(i) for i in x.split(',')])
        )

    def label_items(self) -> None:
        """
        Labels dataset entries as appropriate (1) if all items exceed the frequency threshold,
        otherwise labels them as inappropriate (0).
        """
        logging.info("Labeling items based on frequency...")
        exploded_items = self.dataset['processed_items'].str.split(', ').explode()
        item_counts = exploded_items.value_counts()
        total_entries = len(self.dataset)

        def label_item(item: str) -> int:
            freq = item_counts.get(item, 0) / total_entries
            return 1 if freq >= self.frequency_threshold else 0

        self.dataset['item_labels'] = self.dataset['processed_items'].apply(
            lambda x: [label_item(i) for i in x.split(', ')]
        )
        self.dataset['label'] = self.dataset['item_labels'].apply(
            lambda labels: 1 if all(l == 1 for l in labels) else 0
        )

    def vectorize_features(self) -> None:
        """
        Creates combined feature vectors using TF-IDF and SentenceTransformer embeddings.
        """
        logging.info("Vectorizing features using TF-IDF and embeddings...")
        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.dataset['processed_items'])
        # Embedding generation
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        embeddings = self.embedding_model.encode(self.dataset['processed_items'].tolist(), convert_to_tensor=False)
        self.embeddings_sparse = csr_matrix(embeddings)
        # Combine features
        self.combined_features = hstack([self.tfidf_matrix, self.embeddings_sparse])

    def fit_scaler_and_transform(self) -> None:
        """
        Fits a StandardScaler to the combined features and transforms them.
        """
        logging.info("Fitting scaler and transforming combined features...")
        self.scaler = StandardScaler(with_mean=False)
        # Note: Scaling a sparse matrix with StandardScaler (with_mean=False) is allowed,
        # but the output may be dense.
        self.combined_features = self.scaler.fit_transform(self.combined_features)

    def train_classifier(self) -> None:
        """
        Trains a Ridge regression model to predict the appropriateness label.
        """
        logging.info("Training Ridge Regression classifier...")
        X = self.combined_features
        y = self.dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.classifier = Ridge(alpha=1.0)
        self.classifier.fit(X_train, y_train)
        # Evaluate the classifier
        y_pred = self.classifier.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Classifier Evaluation - MSE: {mse:.4f}, R^2: {r2:.4f}")
        self.classifier_metrics = {'mse': mse, 'r2': r2}

    def fit_nearest_neighbors(self) -> None:
        """
        Fits a Nearest Neighbors model using appropriate items for later similarity adjustments.
        """
        logging.info("Fitting Nearest Neighbors model on appropriate items...")
        appropriate_idx = self.dataset[self.dataset['label'] == 1].index
        if len(appropriate_idx) == 0:
            logging.warning("No appropriate items available for Nearest Neighbors fitting.")
            self.nn_model = None
            return
        appropriate_features = self.combined_features[appropriate_idx]
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn_model.fit(appropriate_features)

    def fit_density_model(self) -> None:
        """
        Fits a Local Outlier Factor (LOF) model on appropriate items for outlier detection.
        """
        logging.info("Fitting Local Outlier Factor model on appropriate items...")
        appropriate_idx = self.dataset[self.dataset['label'] == 1].index
        if len(appropriate_idx) == 0:
            logging.warning("No appropriate items available for density model fitting.")
            self.lof_model = None
            return
        appropriate_features = self.combined_features[appropriate_idx]
        self.lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        self.lof_model.fit(appropriate_features)

    def prepare_embedding_matrix(self) -> None:
        """
        Prepares an embedding matrix from the processed items for cosine similarity checks.
        """
        logging.info("Preparing embedding matrix for similarity calculations...")
        self.embedding_matrix = self.embedding_model.encode(
            self.dataset['processed_items'].tolist(), convert_to_tensor=False
        )

    def get_conditional_probability(self, item: str) -> float:
        """
        Retrieves the conditional probability of an item (given the context) from the OpenAI LLM.
        This probability is used as an additional signal (veto) in the decision process.

        Args:
            item (str): The item text.

        Returns:
            float: The estimated probability (0.0 if an error occurs).
        """
        try:
            prompt = self.context + item
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                logprobs=5,
                echo=False,
                temperature=0
            )
            token_info = response['choices'][0]['logprobs']
            if 'token_logprobs' in token_info and token_info['token_logprobs']:
                log_prob = token_info['token_logprobs'][0]
                probability = np.exp(log_prob)
                return probability
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error for item '{item}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error in get_conditional_probability for '{item}': {e}")
        return 0.0

    def is_outlier(self, feature: csr_matrix) -> bool:
        """
        Determines whether the given feature vector is an outlier as per the LOF model.

        Args:
            feature (csr_matrix): The feature vector.

        Returns:
            bool: True if the feature is an outlier, False otherwise.
        """
        if self.lof_model is None:
            return False
        prediction = self.lof_model.predict(feature)
        return prediction[0] == -1

    def get_features(self, text: str) -> csr_matrix:
        """
        Generates a combined (TF-IDF + embedding) feature vector for the given text,
        and scales it using the trained scaler.

        Args:
            text (str): Input text.

        Returns:
            csr_matrix: Scaled feature vector.
        """
        processed = preprocess_item(text)
        tfidf_vector = self.tfidf.transform([processed])
        embedding = self.embedding_model.encode([processed], convert_to_tensor=False)
        embedding_sparse = csr_matrix(embedding)
        combined = hstack([tfidf_vector, embedding_sparse])
        combined_scaled = self.scaler.transform(combined)
        return combined_scaled

    def evaluate_item(self, new_item: str) -> str:
        """
        Evaluates whether a single new item is appropriate.

        The decision is based on four signals:
          - The classifier's predicted score (must be ≥ score_threshold)
          - The LLM conditional probability (must be ≥ llm_probability_threshold)
          - Cosine similarity to known appropriate items (must be ≥ similarity_threshold)
          - Not being flagged as an outlier by the LOF model

        Args:
            new_item (str): The item to evaluate.

        Returns:
            str: "Appropriate" or "Inappropriate"
        """
        if not new_item or not isinstance(new_item, str):
            logging.error("Invalid item provided for evaluation.")
            return "Inappropriate"

        logging.info(f"Evaluating new item: '{new_item}'")
        feature = self.get_features(new_item)
        predicted_score = self.classifier.predict(feature)[0]
        logging.info(f"Classifier predicted score: {predicted_score:.4f}")

        # Retrieve LLM conditional probability
        llm_probability = self.get_conditional_probability(new_item)
        logging.info(f"LLM conditional probability: {llm_probability:.4f}")

        # Cosine similarity between new item embedding and training embeddings
        new_embedding = self.embedding_model.encode(
            [preprocess_item(new_item)], convert_to_tensor=False
        )
        similarities = cosine_similarity(new_embedding, self.embedding_matrix)
        max_similarity = np.max(similarities)
        logging.info(f"Maximum cosine similarity: {max_similarity:.4f}")

        # Density-based outlier detection
        outlier = self.is_outlier(feature)
        if outlier:
            logging.info("Item detected as an outlier based on density model.")

        # Decision: all signals must meet thresholds
        if (predicted_score >= self.score_threshold and
            llm_probability >= self.llm_probability_threshold and
            max_similarity >= self.similarity_threshold and
            not outlier):
            decision = "Appropriate"
        else:
            decision = "Inappropriate"

        logging.info(f"Final decision for '{new_item}': {decision}")
        return decision

    def evaluate_items(self, new_items: List[str]) -> List[Tuple[str, str]]:
        """
        Evaluates a list of items.

        Args:
            new_items (List[str]): A list of item strings.

        Returns:
            List[Tuple[str, str]]: A list of tuples with each item and its evaluation.
        """
        results = []
        for item in new_items:
            decision = self.evaluate_item(item)
            results.append((item, decision))
        return results


# --------------------- #
#        Main           #
# --------------------- #

def main():
    """
    Main function to run the appropriateness evaluation system.
    """
    # Securely prompt for the OpenAI API key
    api_key = getpass("Please enter your OpenAI API key: ").strip()
    if not api_key:
        logging.error("No API key provided. Exiting.")
        return
    openai.api_key = api_key

    # Get context from user
    context = input("Enter the context description (e.g., 'List common ingredients in a pasta recipe: '): ").strip()
    if not context.endswith(': '):
        context = context.rstrip() + ': '

    # Optionally load a custom dataset
    use_custom_dataset = input("Do you want to provide a custom dataset? (y/n): ").strip().lower()
    if use_custom_dataset == 'y':
        dataset_path = input("Enter the path to your dataset CSV file (with columns 'item_id' and 'items'): ").strip()
        if not os.path.exists(dataset_path):
            logging.error("Dataset file not found. Exiting.")
            return
        try:
            dataset = pd.read_csv(dataset_path)
        except Exception as e:
            logging.error(f"Error reading dataset: {e}")
            return
        if 'item_id' not in dataset.columns or 'items' not in dataset.columns:
            logging.error("Dataset must contain 'item_id' and 'items' columns. Exiting.")
            return
    else:
        dataset = None  # Use sample dataset

    # Get items to evaluate
    items_input = input("Enter the items you want to evaluate, separated by commas: ").strip()
    if not items_input:
        logging.error("No items provided for evaluation. Exiting.")
        return
    target_items = [item.strip() for item in items_input.split(',') if item.strip()]

    # Initialize and run the evaluator
    evaluator = AppropriatenessEvaluator(
        context=context,
        dataset=dataset,
        item_column='items',
        frequency_threshold=0.4,
        similarity_threshold=0.7,
        score_threshold=0.5,
        llm_probability_threshold=0.01,
        model_name="text-davinci-003",
        embedding_model_name='all-MiniLM-L6-v2'
    )

    results = evaluator.evaluate_items(target_items)
    print("\n=== Item Evaluation Results ===")
    for item, decision in results:
        print(f"{item}: {decision}")


if __name__ == "__main__":
    main()
