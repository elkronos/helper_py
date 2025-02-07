#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced AppropriatenessEvaluator

This module evaluates whether given items (e.g., ingredients) are “appropriate” in a particular context.
It uses a combination of TF–IDF, SentenceTransformer embeddings, a Ridge regression classifier,
LLM-based conditional probability, nearest neighbors similarity and LOF–based density detection.
"""

import os
import re
import logging
from typing import List, Tuple, Optional

import numpy as np
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
from getpass import getpass

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
#   Preprocessing Function  #
# --------------------- #

# Create global instances for stopwords and lemmatizer to avoid reinitialization on each call.
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def preprocess_item(item: str) -> str:
    """
    Preprocess a text item: lowercase, remove special characters, tokenize, remove stopwords, and lemmatize.
    
    Args:
        item (str): Input text.
    
    Returns:
        str: Cleaned and preprocessed text.
    """
    if not item or not item.strip():
        return ""
    # Lowercase
    item = item.lower()
    # Remove special characters (retain alphanumerics and spaces)
    item = re.sub(r'[^a-z0-9\s]', '', item)
    # Tokenize and remove extra whitespace
    tokens = item.split()
    # Remove stopwords and lemmatize each token
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS]
    return ' '.join(tokens)


# --------------------- #
#   AppropriatenessEvaluator Class  #
# --------------------- #

class AppropriatenessEvaluator:
    def __init__(
        self,
        context: str,
        dataset: Optional[pd.DataFrame] = None,
        item_column: str = 'items',
        frequency_threshold: float = 0.4,
        similarity_threshold: float = 0.7,
        score_threshold: float = 0.5,
        llm_probability_threshold: float = 0.1,
        model_name: str = "text-davinci-003",
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initializes the evaluator.
        
        Args:
            context (str): Context description.
            dataset (Optional[pd.DataFrame]): DataFrame with items. If None, a sample dataset is created.
            item_column (str): Column name containing items.
            frequency_threshold (float): Proportion threshold (per entry) for labeling an item as appropriate.
            similarity_threshold (float): Minimum cosine similarity for similarity-based adjustment.
            score_threshold (float): Classifier decision threshold.
            llm_probability_threshold (float): Minimum acceptable LLM probability.
            model_name (str): OpenAI model to use.
            embedding_model_name (str): SentenceTransformer model name.
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

        # Build the pipeline
        self.preprocess_data()
        self.label_items()
        self.vectorize_features()
        self.train_classifier()
        self.fit_nearest_neighbors()
        self.fit_density_model()
        self.fit_scaler()
        self.prepare_embedding_matrix()

    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Creates a sample dataset.
        
        Returns:
            pd.DataFrame: DataFrame with sample items.
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

    def preprocess_data(self):
        """
        Preprocesses items in the dataset.
        """
        logging.info("Preprocessing dataset items...")
        # Split items on comma (regardless of spacing), strip, preprocess, then rejoin with ', '
        self.dataset['processed_items'] = self.dataset[self.item_column].apply(
            lambda x: ', '.join([preprocess_item(i.strip()) for i in x.split(',') if i.strip()])
        )

    def label_items(self):
        """
        Labels each dataset entry as appropriate (1) or inappropriate (0) based on item frequency.
        An entry is labeled appropriate only if every preprocessed item is frequent enough.
        """
        logging.info("Labeling items based on frequency...")
        # Explode items into individual tokens
        all_items = self.dataset['processed_items'].str.split(', ').explode()
        # Count how many entries contain each item (frequency relative to number of entries)
        item_counts = all_items.value_counts()
        total_entries = len(self.dataset)

        def label_item(item: str) -> int:
            freq = item_counts.get(item, 0) / total_entries
            return 1 if freq >= self.frequency_threshold else 0

        # Label items for each entry; if any item is below threshold, mark the entry as inappropriate.
        self.dataset['item_labels'] = self.dataset['processed_items'].apply(
            lambda x: [label_item(i) for i in x.split(', ') if i]
        )
        self.dataset['label'] = self.dataset['item_labels'].apply(lambda labels: 1 if all(label == 1 for label in labels) else 0)

    def vectorize_features(self):
        """
        Vectorizes the processed items using TF–IDF and SentenceTransformer embeddings,
        then combines them into a single feature matrix.
        """
        logging.info("Vectorizing features with TF–IDF...")
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.dataset['processed_items'])

        logging.info("Generating dense embeddings using SentenceTransformer...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        embeddings = self.embedding_model.encode(self.dataset['processed_items'].tolist(), convert_to_tensor=False)
        # Convert dense embeddings to sparse matrix to enable hstack with tfidf_matrix.
        self.embeddings_sparse = csr_matrix(embeddings)

        logging.info("Combining TF–IDF features with embeddings...")
        self.combined_features = hstack([self.tfidf_matrix, self.embeddings_sparse])

    def train_classifier(self):
        """
        Trains a Ridge regression classifier on the combined features.
        """
        logging.info("Training classifier...")
        X = self.combined_features
        y = self.dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier = Ridge(alpha=1.0)
        self.classifier.fit(X_train, y_train)

        # Evaluate performance and log statistics.
        y_pred = self.classifier.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Classifier Evaluation: MSE = {mse:.4f}, R² = {r2:.4f}")
        self.classifier_metrics = {'mse': mse, 'r2': r2}

    def fit_nearest_neighbors(self):
        """
        Fits a Nearest Neighbors model on the features of entries labeled as appropriate.
        """
        logging.info("Fitting Nearest Neighbors model...")
        appropriate_indices = self.dataset.index[self.dataset['label'] == 1].tolist()
        if not appropriate_indices:
            logging.warning("No appropriate items found; skipping Nearest Neighbors fitting.")
            self.nn_model = None
            return

        appropriate_features = self.combined_features[appropriate_indices]
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn_model.fit(appropriate_features)

    def fit_density_model(self):
        """
        Fits a Local Outlier Factor model on features of appropriate items for density-based outlier detection.
        """
        logging.info("Fitting Local Outlier Factor (LOF) model...")
        appropriate_indices = self.dataset.index[self.dataset['label'] == 1].tolist()
        if not appropriate_indices:
            logging.warning("No appropriate items found; skipping LOF fitting.")
            self.lof_model = None
            return

        appropriate_features = self.combined_features[appropriate_indices]
        # LOF with novelty=True to allow prediction on new data.
        self.lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        # LOF requires a dense array; if combined_features is sparse, convert it.
        if hasattr(appropriate_features, "toarray"):
            appropriate_features = appropriate_features.toarray()
        self.lof_model.fit(appropriate_features)

    def fit_scaler(self):
        """
        Fits a StandardScaler to the combined features for normalization.
        """
        logging.info("Fitting StandardScaler on combined features...")
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(self.combined_features)

    def prepare_embedding_matrix(self):
        """
        Prepares a dense embedding matrix for similarity calculations.
        """
        logging.info("Preparing embedding matrix for similarity calculations...")
        self.embedding_matrix = np.array(
            self.embedding_model.encode(self.dataset['processed_items'].tolist(), convert_to_tensor=False)
        )

    def get_conditional_probability(self, item: str) -> float:
        """
        Retrieves the conditional probability of an item given the context using the OpenAI API.
        This implementation is experimental and may require further tuning.
        
        Args:
            item (str): The item to evaluate.
        
        Returns:
            float: Estimated probability (0.0 if not obtainable).
        """
        try:
            prompt = f"{self.context} {item}"
            # Use max_tokens=1 to get a minimal completion and obtain logprobs.
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                logprobs=100,
                echo=True,
                temperature=0
            )
            log_probs = response['choices'][0]['logprobs']['token_logprobs']
            tokens = response['choices'][0]['logprobs']['tokens']
            if not tokens:
                return 0.0
            # Compare the last token (after stripping) with the lowercased item.
            last_token = tokens[-1].strip().lower()
            if last_token == item.lower():
                return np.exp(log_probs[-1])
            else:
                return 0.0
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error for item '{item}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error retrieving probability for '{item}': {e}")
        return 0.0

    def is_outlier(self, feature: csr_matrix) -> bool:
        """
        Checks if a feature vector is an outlier based on the LOF model.
        
        Args:
            feature (csr_matrix): Feature vector.
        
        Returns:
            bool: True if an outlier; False otherwise.
        """
        if self.lof_model is None:
            return False
        try:
            # LOF requires a dense array.
            feature_dense = feature.toarray() if hasattr(feature, "toarray") else feature
            prediction = self.lof_model.predict(feature_dense)
            return prediction[0] == -1
        except Exception as e:
            logging.error(f"Error in outlier detection: {e}")
            return False

    def evaluate_item(self, new_item: str) -> str:
        """
        Evaluates a single new item.
        
        Args:
            new_item (str): The item to evaluate.
        
        Returns:
            str: "Appropriate" or "Inappropriate".
        """
        if not new_item or not new_item.strip():
            logging.warning("Empty item provided for evaluation.")
            return "Inappropriate"

        logging.info(f"Evaluating item: '{new_item}'")
        try:
            # Preprocess the item.
            processed = preprocess_item(new_item)
            if not processed:
                logging.warning("Item is empty after preprocessing.")
                return "Inappropriate"

            # Vectorize using TF–IDF.
            tfidf_vector = self.tfidf.transform([processed])
            # Obtain dense embedding and convert to sparse.
            embedding = self.embedding_model.encode([processed], convert_to_tensor=False)
            embedding_sparse = csr_matrix(embedding)
            # Combine features.
            combined_feature = hstack([tfidf_vector, embedding_sparse])
            # Normalize features.
            combined_feature = self.scaler.transform(combined_feature)

            # Classifier prediction (note: classifier was trained without LLM probability).
            predicted_score = self.classifier.predict(combined_feature)[0]
            decision = "Appropriate" if predicted_score >= self.score_threshold else "Inappropriate"
            logging.info(f"Classifier predicted score: {predicted_score:.4f} (threshold: {self.score_threshold})")

            # LLM conditional probability check.
            llm_probability = self.get_conditional_probability(new_item)
            logging.info(f"LLM probability for '{new_item}': {llm_probability:.4f}")
            if llm_probability < self.llm_probability_threshold:
                logging.info("Decision overridden due to low LLM probability.")
                decision = "Inappropriate"

            # Similarity-based adjustment.
            ingredient_embedding = np.array(embedding)  # shape: (1, d)
            similarities = cosine_similarity(ingredient_embedding, self.embedding_matrix)
            max_similarity = np.max(similarities)
            logging.info(f"Maximum cosine similarity: {max_similarity:.4f}")
            if max_similarity < self.similarity_threshold and predicted_score >= self.score_threshold:
                logging.info("Decision adjusted due to low similarity with known appropriate items.")
                decision = "Inappropriate"

            # Density-based outlier check.
            if self.is_outlier(combined_feature):
                logging.info("Decision adjusted based on density-based outlier detection.")
                decision = "Inappropriate"

            logging.info(f"Final decision for '{new_item}': {decision}")
            return decision

        except Exception as e:
            logging.error(f"Error evaluating item '{new_item}': {e}")
            return "Evaluation Error"

    def evaluate_items(self, new_items: List[str]) -> List[Tuple[str, str]]:
        """
        Evaluates a list of new items.
        
        Args:
            new_items (List[str]): Items to evaluate.
        
        Returns:
            List[Tuple[str, str]]: Tuples of (item, decision).
        """
        results = []
        for item in new_items:
            decision = self.evaluate_item(item)
            results.append((item, decision))
        return results


# --------------------- #
#         Main Function         #
# --------------------- #

def main():
    """
    Main execution function for evaluating item appropriateness.
    """
    # Securely obtain OpenAI API key.
    api_key = getpass("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    # Prompt for context.
    context = input("Enter the context description (e.g., 'List common ingredients in a pasta recipe'): ").strip()
    if not context.endswith(':'):
        context += ':'

    # Option to load a custom dataset.
    use_custom_dataset = input("Do you want to provide a custom dataset? (y/n): ").strip().lower()
    if use_custom_dataset == 'y':
        dataset_path = input("Enter the path to your dataset CSV file (with columns 'item_id' and 'items'): ").strip()
        if not os.path.exists(dataset_path):
            logging.error("Dataset file not found. Exiting.")
            return
        dataset = pd.read_csv(dataset_path)
        if 'item_id' not in dataset.columns or 'items' not in dataset.columns:
            logging.error("Dataset must contain 'item_id' and 'items' columns. Exiting.")
            return
    else:
        dataset = None

    # Prompt for items to evaluate.
    items_input = input("\nEnter the items you want to evaluate, separated by commas (e.g., 'Cheetos, chocolate, tofu'): ").strip()
    target_items = [item.strip() for item in items_input.split(',') if item.strip()]
    if not target_items:
        logging.error("No valid items provided for evaluation. Exiting.")
        return

    # Initialize the evaluator.
    evaluator = AppropriatenessEvaluator(
        context=context,
        dataset=dataset,
        item_column='items',
        frequency_threshold=0.4,       # Adjust as needed.
        similarity_threshold=0.7,      # Adjust as needed.
        score_threshold=0.5,           # Adjust as needed.
        llm_probability_threshold=0.1, # Adjust as needed.
        model_name="text-davinci-003",
        embedding_model_name='all-MiniLM-L6-v2'
    )

    # Evaluate provided items.
    logging.info("Starting evaluation of provided items...")
    results = evaluator.evaluate_items(target_items)
    print("\n=== Evaluation Results ===")
    for item, decision in results:
        print(f"{item}: {decision}")


if __name__ == "__main__":
    main()
