import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

# Ensure Pandas shows all columns
pd.set_option('display.max_columns', None)

# Load dataset with sarcastic reviews
df_sarcastic_reviews = pd.read_csv("data/Sarcastic_Reviews.csv")

# Extract review text and corresponding labels (assumed ground truth)
X_texts = df_sarcastic_reviews["Text"]
y_true = np.where(df_sarcastic_reviews["Score"] >= 4, 1, 0)  # 1 for positive, 0 for negative

# Store model results dynamically
model_results = []

# ======== VADER Sentiment Analysis ========
vader = SentimentIntensityAnalyzer()
y_pred_vader = [1 if vader.polarity_scores(text)['compound'] > 0 else 0 for text in X_texts]
vader_report = classification_report(y_true, y_pred_vader, output_dict=True)

# Append results dynamically
model_results.append({
    "Model": "VADER",
    "Detected as Positive (%)": sum(y_pred_vader) / len(y_pred_vader) * 100,
    "Detected as Negative (%)": (1 - sum(y_pred_vader) / len(y_pred_vader)) * 100,
    "Misclassification Rate": round((1 - vader_report["accuracy"]) * 100, 2)
})


# ============== Naïve Bayes Sentiment Analysis =====================
# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_texts)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_true), y=y_true)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train a simple Naïve Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, y_true)

# Predict using Naïve Bayes
y_pred_nb = nb_classifier.predict(X_tfidf)
nb_report = classification_report(y_true, y_pred_nb, output_dict=True)

# Append results dynamically
model_results.append({
    "Model": "Naïve Bayes",
    "Detected as Positive (%)": sum(y_pred_nb) / len(y_pred_nb) * 100,
    "Detected as Negative (%)": (1 - sum(y_pred_nb) / len(y_pred_nb)) * 100,
    "Misclassification Rate": round((1 - nb_report["accuracy"]) * 100, 2)
})

# ============================= LSTM Sentiment Analysis =====================
# Load tokenizer to check token length
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def safe_predict(text, model_pipeline, tokenizer):
    """Truncate text properly before passing it to the model"""
    encoded = tokenizer(text, truncation=True, max_length=512)

    # Extract the truncated text (reconstruct from tokens)
    truncated_text = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    # Pass truncated text to the pipeline
    return model_pipeline(truncated_text)[0]['label']


lstm_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Apply safe predictions
y_pred_lstm = [1 if safe_predict(text, lstm_sentiment, tokenizer) == "POSITIVE" else 0 for text in X_texts]

lstm_report = classification_report(y_true, y_pred_lstm, output_dict=True)
print(classification_report(y_true, y_pred_lstm, output_dict=True))
# Append results dynamically
model_results.append({
    "Model": "LSTM",
    "Detected as Positive (%)": sum(y_pred_lstm) / len(y_pred_lstm) * 100,
    "Detected as Negative (%)": (1 - sum(y_pred_lstm) / len(y_pred_lstm)) * 100,
    "Misclassification Rate": round((1 - lstm_report["accuracy"]) * 100, 2)
})

# ============================= BERT Sentiment Analysis =====================
bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def safe_predict_bert(text, model_pipeline, tokenizer):
    """Ensures text does not exceed 512 tokens before passing to the BERT model."""

    # Tokenize input and enforce truncation
    encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

    # Extract tokenized input (ensure it's properly formatted)
    model_input = tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)[0]

    # Pass truncated input to the model
    return model_pipeline(model_input, truncation=True)[0]['label']


# Apply safe predictions
y_pred_bert = [1 if any(star in safe_predict_bert(text, bert_sentiment, tokenizer) for star in ["4", "5"]) else 0 for
               text in X_texts]

bert_report = classification_report(y_true, y_pred_bert, output_dict=True)
print(classification_report(y_true, y_pred_bert, output_dict=True))
# Append results dynamically
model_results.append({
    "Model": "BERT",
    "Detected as Positive (%)": sum(y_pred_bert) / len(y_pred_bert) * 100,
    "Detected as Negative (%)": (1 - sum(y_pred_bert) / len(y_pred_bert)) * 100,
    "Misclassification Rate": round((1 - bert_report["accuracy"]) * 100, 2)
})

# ============================= Misclassification Calculation Using Confusion Matrix =====================
def calculate_metrics(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    detected_positive = (tp + fp) / (tp + fp + tn + fn) * 100  # Total times model predicted positive
    detected_negative = (tn + fn) / (tp + fp + tn + fn) * 100  # Total times model predicted negative
    misclassification_rate = (fp + fn) / (tp + fp + tn + fn) * 100  # Incorrect predictions

    return round(detected_positive, 2), round(detected_negative, 2), round(misclassification_rate, 2)


# Apply confusion matrix metrics
for model_name, y_pred in zip(["VADER", "Naïve Bayes", "LSTM", "BERT"],
                              [y_pred_vader, y_pred_nb, y_pred_lstm, y_pred_bert]):
    detected_positive, detected_negative, misclassification_rate = calculate_metrics(y_pred, y_true)

    model_results.append({
        "Model": model_name,
        "Detected as Positive (%)": detected_positive,
        "Detected as Negative (%)": detected_negative,
        "Misclassification Rate": misclassification_rate
    })

# Convert results to DataFrame
df_misclassification = pd.DataFrame(model_results)

# Display the final misclassification table
print(df_misclassification)
