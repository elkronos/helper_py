# This is an attempt to convert this script into python and improve it:
# https://github.com/elkronos/public_examples/blob/main/analysis/sentiment_negation_intensifiers.R

import string
import pandas as pd
from afinn import Afinn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Initialize Afinn for word-level sentiment analysis
afinn = Afinn()

# Load pre-trained BERT model and tokenizer for sentence-level sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text, level='sentence'):
    if level == 'sentence':
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = softmax(outputs.logits.detach().numpy(), axis=1)
        return (scores[:, 4] - scores[:, 0]).item()  # Difference between positive and negative scores
    elif level == 'word':
        return afinn.score(text)

def process_text(text, sentiment_level='sentence'):
    words = text.split()
    processed_words = []

    for word in words:
        word_clean = word.strip(string.punctuation).lower()  # Strip punctuation and convert to lower case
        sentiment_score = analyze_sentiment(word_clean, level=sentiment_level)

        emotion_lexicon = {
            "happy": "joy", "sad": "sadness",
            "amazing": "surprise", "terrible": "anger",
            # Add more words as needed
        }
        emotion = emotion_lexicon.get(word_clean, None)

        processed_words.append({
            "word": word,
            "sentiment_score": sentiment_score,
            "emotion": emotion
        })
    return processed_words

def detect_sentiments_and_emotions(df, columns, sentiment_level='sentence'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    processed_data = []

    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        for index, text in df[column].items():
            processed_words = process_text(text, sentiment_level)
            for word in processed_words:
                word['text_index'] = index
                word['source_column'] = column
            processed_data.extend(processed_words)

    return pd.DataFrame(processed_data)

# Example usage
data = {
    "text_column1": ["I am not unhappy about this amazing event.", "This is a very bad situation."],
    "text_column2": ["She is hardly ever sad.", "It's rarely that simple."]
}

df = pd.DataFrame(data)
result_df = detect_sentiments_and_emotions(df, ["text_column1", "text_column2"])
print(result_df)
