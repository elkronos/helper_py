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
from typing import List, Tuple, Optional

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
#    AppropriatenessEvaluator Class #
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
        model_name: str = "text-davinci-003",
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initializes the AppropriatenessEvaluator with the given parameters.

        Args:
            context (str): The context description.
            dataset (Optional[pd.DataFrame]): The dataset containing items. If None, a sample dataset is created.
            item_column (str): The column name containing items in the dataset.
            frequency_threshold (float): Minimum frequency to label an item as appropriate.
            similarity_threshold (float): Minimum cosine similarity for similarity-based adjustment.
            score_threshold (float): Threshold for classifier decision.
            model_name (str): The OpenAI LLM model to use.
            embedding_model_name (str): The SentenceTransformer model to use for embeddings.
        """
        self.context = context
        self.dataset = dataset if dataset is not None else self.create_sample_dataset()
        self.item_column = item_column
        self.frequency_threshold = frequency_threshold
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        # Initialize components
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
        Creates a sample dataset based on the provided context.

        Returns:
            pd.DataFrame: DataFrame containing item IDs and their descriptions.
        """
        logging.info("Creating a sample dataset...")
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

    def preprocess_data(self):
        """
        Preprocesses the items in the dataset.
        """
        logging.info("Preprocessing items...")
        # Apply preprocessing to each item list
        self.dataset['processed_items'] = self.dataset[self.item_column].apply(
            lambda x: ', '.join([preprocess_item(i) for i in x.split(',')])
        )

    def label_items(self):
        """
        Labels items as appropriate (1) or inappropriate (0) based on their frequency.
        """
        logging.info("Labeling items based on frequency...")
        # Explode the items into individual rows
        all_items = self.dataset['processed_items'].str.split(', ').explode()
        # Calculate item frequencies
        item_counts = all_items.value_counts()
        total_entries = len(self.dataset)
        # Define label function
        def label_item(item: str) -> int:
            freq = item_counts.get(item, 0) / total_entries
            return 1 if freq >= self.frequency_threshold else 0  # 1: appropriate, 0: inappropriate
        
        # Apply labeling to each item in each entry
        # Here, we label the entire entry as inappropriate if any item is inappropriate
        self.dataset['item_labels'] = self.dataset['processed_items'].apply(
            lambda x: [label_item(i) for i in x.split(', ')]
        )
        # Define entry label
        self.dataset['label'] = self.dataset['item_labels'].apply(lambda labels: 0 if 0 in labels else 1)

    def vectorize_features(self):
        """
        Vectorizes the preprocessed items using TF-IDF and word embeddings.
        """
        logging.info("Vectorizing features using TF-IDF...")
        # Initialize TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer()
        # Fit and transform the processed items
        self.tfidf_matrix = self.tfidf.fit_transform(self.dataset['processed_items'])
        
        logging.info("Generating embeddings using SentenceTransformer...")
        # Initialize SentenceTransformer for embeddings
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        # Generate embeddings
        embeddings = self.embedding_model.encode(self.dataset['processed_items'].tolist(), convert_to_tensor=False)
        
        logging.info("Combining TF-IDF vectors with embeddings...")
        # Convert embeddings to sparse matrix
        self.embeddings_sparse = csr_matrix(embeddings)
        # Combine features
        self.combined_features = hstack([self.tfidf_matrix, self.embeddings_sparse])

    def train_classifier(self):
        """
        Trains a Ridge Regression model to predict appropriateness scores.
        """
        logging.info("Training classifier...")
        X = self.combined_features
        y = self.dataset['label']
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Initialize and train model
        self.classifier = Ridge(alpha=1.0)
        self.classifier.fit(X_train, y_train)
        # Evaluate
        logging.info("Evaluating classifier on test set...")
        y_pred = self.classifier.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Classifier Evaluation - Mean Squared Error: {mse:.4f}, R^2 Score: {r2:.4f}")

    def fit_nearest_neighbors(self):
        """
        Fits a Nearest Neighbors model on appropriate items for similarity-based adjustments.
        """
        logging.info("Fitting Nearest Neighbors model on appropriate items...")
        # Extract appropriate items
        appropriate_indices = self.dataset[self.dataset['label'] == 1].index
        appropriate_features = self.combined_features[appropriate_indices]
        
        # Initialize and fit NearestNeighbors
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn_model.fit(appropriate_features)

    def fit_density_model(self):
        """
        Fits a Local Outlier Factor model on appropriate items for density-based outlier detection.
        """
        logging.info("Fitting Local Outlier Factor model on appropriate items...")
        # Extract appropriate items
        appropriate_indices = self.dataset[self.dataset['label'] == 1].index
        appropriate_features = self.combined_features[appropriate_indices]
        
        # Initialize and fit LOF
        self.lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        self.lof_model.fit(appropriate_features)

    def fit_scaler(self):
        """
        Fits a StandardScaler on the combined features for normalization.
        """
        logging.info("Fitting StandardScaler on combined features...")
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(self.combined_features)

    def prepare_embedding_matrix(self):
        """
        Prepares the embedding matrix for similarity calculations.
        """
        logging.info("Preparing embedding matrix for similarity calculations...")
        self.embedding_matrix = self.embedding_model.encode(self.dataset['processed_items'].tolist(), convert_to_tensor=False)

    def get_conditional_probability(self, item: str) -> float:
        """
        Retrieves the conditional probability of an item given the context from the LLM.

        Args:
            item (str): The item to get the probability for.

        Returns:
            float: The conditional probability of the item. Returns 0.0 if not found.
        """
        try:
            prompt = self.context + item
            response = openai.Completion.create(
                model=self.model_name,
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

    def is_outlier(self, feature: csr_matrix) -> bool:
        """
        Determines if a feature vector is an outlier.

        Args:
            feature (csr_matrix): Feature vector.

        Returns:
            bool: True if outlier, False otherwise.
        """
        return self.lof_model.predict(feature) == -1

    def evaluate_item(
        self,
        new_item: str
    ) -> str:
        """
        Evaluates whether a new item is appropriate within the given context.

        Args:
            new_item (str): The item to evaluate.

        Returns:
            str: "Appropriate" or "Inappropriate"
        """
        logging.info(f"Evaluating new item: '{new_item}'")
        # Preprocess
        processed = preprocess_item(new_item)
        
        # TF-IDF Vectorization
        tfidf_vector = self.tfidf.transform([processed])
        
        # Embedding
        embedding = self.embedding_model.encode([processed], convert_to_tensor=False)
        embedding_sparse = csr_matrix(embedding)
        
        # Combine Features
        combined_feature = hstack([tfidf_vector, embedding_sparse])
        
        # Normalize if scaler is provided
        if self.scaler:
            combined_feature = self.scaler.transform(combined_feature)
        
        # LLM Conditional Probability
        llm_probability = self.get_conditional_probability(new_item)
        # Convert probability to log probability
        llm_log_prob = np.log(llm_probability + 1e-10)  # Add epsilon to avoid log(0)
        llm_log_prob_feature = csr_matrix([llm_log_prob])
        
        # Final Combined Feature (Appending LLM log probability)
        final_feature = hstack([combined_feature, llm_log_prob_feature])
        
        # Predict using classifier
        predicted_score = self.classifier.predict(final_feature)[0]
        
        # Initial Decision based on classifier score
        decision = "Appropriate" if predicted_score >= self.score_threshold else "Inappropriate"
        
        # Similarity Adjustment
        ingredient_embedding = embedding
        similarities = cosine_similarity(ingredient_embedding, self.embedding_matrix)
        max_similarity = np.max(similarities)
        logging.info(f"Maximum cosine similarity: {max_similarity:.4f}")
        
        if max_similarity < self.similarity_threshold and predicted_score >= self.score_threshold:
            decision = "Inappropriate"
            logging.info("Adjusted decision based on low similarity to known appropriate items.")
        
        # Density-Based Outlier Detection
        if self.is_outlier(final_feature):
            decision = "Inappropriate"
            logging.info("Adjusted decision based on density-based outlier detection.")
        
        logging.info(f"Final Decision for '{new_item}': {decision}")
        return decision

    def evaluate_items(self, new_items: List[str]) -> List[Tuple[str, str]]:
        """
        Evaluates a list of new items for their appropriateness within the given context.

        Args:
            new_items (List[str]): The list of items to evaluate.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing the item and its evaluation.
        """
        results = []
        for item in new_items:
            decision = self.evaluate_item(item)
            results.append((item, decision))
        return results

# --------------------- #
#        Main Function   #
# --------------------- #

def main():
    """
    Main function to execute the automated appropriateness evaluation system.
    """
    # Prompt user for OpenAI API key securely
    api_key = getpass("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

    # Prompt user for context
    context = input("Enter the context description (e.g., 'List common ingredients in a pasta recipe:'): ").strip()
    if not context.endswith(': '):
        context += ': '

    # Option to load custom dataset or use sample
    use_custom_dataset = input("Do you want to provide a custom dataset? (y/n): ").strip().lower()
    if use_custom_dataset == 'y':
        # Prompt user for dataset path or other method to load
        dataset_path = input("Enter the path to your dataset CSV file (with columns 'item_id' and 'items'): ").strip()
        if not os.path.exists(dataset_path):
            logging.error("Dataset file not found. Exiting.")
            return
        dataset = pd.read_csv(dataset_path)
        # Validate dataset columns
        if 'item_id' not in dataset.columns or 'items' not in dataset.columns:
            logging.error("Dataset must contain 'item_id' and 'items' columns. Exiting.")
            return
    else:
        dataset = None  # Use sample dataset

    # Prompt user for items to evaluate
    print("\nEnter the items you want to evaluate, separated by commas (e.g., 'Cheetos, chocolate, tofu'):")
    items_input = input().strip()
    target_items = [item.strip() for item in items_input.split(',') if item.strip()]
    if not target_items:
        logging.error("No items provided for evaluation. Exiting.")
        return

    # Initialize the AppropriatenessEvaluator
    evaluator = AppropriatenessEvaluator(
        context=context,
        dataset=dataset,
        item_column='items',
        frequency_threshold=0.4,      # Can be adjusted as needed
        similarity_threshold=0.7,     # Can be adjusted as needed
        score_threshold=0.5,          # Can be adjusted as needed
        model_name="text-davinci-003",# Ensure this model is accessible
        embedding_model_name='all-MiniLM-L6-v2'  # Can be changed based on preference
    )

    # Evaluate the items
    print("\n=== Item Evaluation Results ===")
    results = evaluator.evaluate_items(target_items)
    for item, decision in results:
        print(f"{item}: {decision}")

# --------------------- #
#       Execute Main     #
# --------------------- #

if __name__ == "__main__":
    main()
