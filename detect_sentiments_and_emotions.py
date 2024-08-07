import string
import pandas as pd
import numpy as np
from afinn import Afinn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from typing import List, Dict, Union
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment", batch_size: int = 32):
        self.afinn = Afinn()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size

    def analyze_sentiment_batch(self, texts: List[str]) -> List[float]:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = softmax(outputs.logits.detach().numpy(), axis=1)
            return (scores[:, 4] - scores[:, 0]).tolist()
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return [0.0] * len(texts)  # Return neutral scores in case of error

    def analyze_sentiment(self, text: str, level: str = 'sentence') -> float:
        try:
            if level == 'sentence':
                return self.analyze_sentiment_batch([text])[0]
            elif level == 'word':
                return self.afinn.score(text)
            else:
                raise ValueError("Invalid sentiment level. Choose 'sentence' or 'word'.")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0  # Return neutral score in case of error

class TextProcessor:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.emotion_lexicon = {
            "happy": "joy", "sad": "sadness", "amazing": "surprise", "terrible": "anger",
            "excited": "joy", "depressed": "sadness", "shocked": "surprise", "furious": "anger",
            "delighted": "joy", "miserable": "sadness", "astonished": "surprise", "irritated": "anger",
            "calm": "peace", "anxious": "fear", "confident": "trust", "disgusted": "disgust"
            # Add more emotion mappings as needed
        }

    def process_text(self, text: str, sentiment_level: str = 'sentence') -> List[Dict[str, Union[str, float, None]]]:
        words = text.split()
        processed_words = []
        for word in words:
            word_clean = word.strip(string.punctuation).lower()
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(word_clean, level=sentiment_level)
            emotion = self.emotion_lexicon.get(word_clean, None)
            processed_words.append({
                "word": word,
                "sentiment_score": sentiment_score,
                "emotion": emotion
            })
        return processed_words

class SentimentEmotionDetector:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor

    def detect_sentiments_and_emotions(self, df: pd.DataFrame, columns: List[str], sentiment_level: str = 'sentence') -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        processed_data = []
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            
            texts = df[column].tolist()
            batch_size = self.text_processor.sentiment_analyzer.batch_size
            
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {column}"):
                batch_texts = texts[i:i+batch_size]
                batch_sentiments = self.text_processor.sentiment_analyzer.analyze_sentiment_batch(batch_texts)
                
                for text, sentiment in zip(batch_texts, batch_sentiments):
                    processed_words = self.text_processor.process_text(text, sentiment_level)
                    for word in processed_words:
                        word['text_index'] = i
                        word['source_column'] = column
                        word['overall_sentiment'] = sentiment
                    processed_data.extend(processed_words)
        
        result_df = pd.DataFrame(processed_data)
        result_df['sentiment_category'] = pd.cut(result_df['sentiment_score'], 
                                                 bins=[-np.inf, -0.5, 0.5, np.inf], 
                                                 labels=['Negative', 'Neutral', 'Positive'])
        return result_df

def main(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment", batch_size: int = 32):
    try:
        # Initialize components
        sentiment_analyzer = SentimentAnalyzer(model_name=model_name, batch_size=batch_size)
        text_processor = TextProcessor(sentiment_analyzer)
        detector = SentimentEmotionDetector(text_processor)

        # Example usage
        data = {
            "text_column1": ["I am not unhappy about this amazing event.", "This is a very bad situation."],
            "text_column2": ["She is hardly ever sad.", "It's rarely that simple."]
        }
        df = pd.DataFrame(data)

        result_df = detector.detect_sentiments_and_emotions(df, ["text_column1", "text_column2"])
        print(result_df)

        # Additional analysis
        print("\nSentiment Distribution:")
        print(result_df['sentiment_category'].value_counts(normalize=True))

        print("\nTop 5 Most Positive Words:")
        print(result_df.nlargest(5, 'sentiment_score')[['word', 'sentiment_score']])

        print("\nTop 5 Most Negative Words:")
        print(result_df.nsmallest(5, 'sentiment_score')[['word', 'sentiment_score']])

        print("\nEmotion Distribution:")
        print(result_df['emotion'].value_counts(normalize=True))

    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()