import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import nltk
from scripts.preprocessing import preprocess_data
import time

file_path = '../data/cleaned_reviews.csv'
data = pd.read_csv(file_path)
# Preprocess data
data = preprocess_data(data)
print(data.columns)
# Download VADER resources
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()


# Predict sentiment using VADER
def vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound > -0.05:
        return 'neutral'
    else:
        return 'negative'


# Start timing
start_time = time.time()
# Apply VADER and evaluate
data['Vader_Sentiment'] = data['Cleaned_Text'].apply(vader_sentiment)
# End timing
end_time = time.time()
# Calculate total processing time
vader_processing_time = end_time - start_time
print(f"VADER Processing Time: {vader_processing_time:.2f} seconds")

data.to_csv('../data/VADER_reviews.csv', index=False)
print('\nVADER Sentiment Analysis Results:')
print(classification_report(data['Sentiment'], data['Vader_Sentiment']))