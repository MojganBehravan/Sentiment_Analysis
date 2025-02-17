import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight



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
# ======== Load Saved XGBoost Model and Vectorizer ========
xgb_vectorizer = joblib.load("data/xgboost_tfidf_vectorizer_tune.pkl")
xgb_model = joblib.load("data/xgboost_sentiment_model_tune.pkl")
#Transform Sarcastic Reviews Using TF-IDF
X_tfidf_xgb = xgb_vectorizer.transform(X_texts)
# Get XGBoost Predictions
y_pred_xgb = xgb_model.predict(X_tfidf_xgb)

#  Ignore neutral (1) by filtering out those instances
mask = (y_true != 1)  # Keep only positive (2) and negative (0) reviews
y_true_filtered = y_true[mask]
y_pred_xgb_filtered = y_pred_xgb[mask]

#  Convert to binary classification (0 = Negative, 2 → 1 = Positive)
y_true_binary = np.where(y_true_filtered == 2, 1, 0)
y_pred_xgb_binary = np.where(y_pred_xgb_filtered == 2, 1, 0)

# Evaluate XGBoost Performance
xgb_report = classification_report(y_true_binary, y_pred_xgb_binary, output_dict=True)
print(classification_report(y_true_binary, y_pred_xgb_binary, zero_division=0))

# Compute statistics
total_samples = len(y_pred_xgb_binary)
positive_percent = (np.count_nonzero(y_pred_xgb_binary == 1) / total_samples) * 100
negative_percent = (np.count_nonzero(y_pred_xgb_binary == 0) / total_samples) * 100
misclassification_rate = round((1 - xgb_report["accuracy"]) * 100, 2)

# Store results dynamically
model_results.append({
    "Model": "XGBoost",
    "Detected as Positive (%)": positive_percent,
    "Detected as Negative (%)": negative_percent,
    "Misclassification Rate": misclassification_rate
})


# ============================= Misclassification Calculation Using Confusion Matrix =====================
def calculate_metrics(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    total_instances = cm.sum()
    correct_predictions = cm.diagonal().sum()  # Sum of correct predictions (diagonal elements)
    misclassified_instances = total_instances - correct_predictions  # Everything else is misclassified

    detected_positive = (sum(cm[:, 1]) / total_instances) * 100 if cm.shape[1] > 1 else (sum(y_pred) / len(
        y_pred)) * 100
    detected_negative = (sum(cm[:, 0]) / total_instances) * 100 if cm.shape[1] > 1 else (1 - sum(y_pred) / len(
        y_pred)) * 100
    misclassification_rate = (misclassified_instances / total_instances) * 100

    return round(detected_positive, 2), round(detected_negative, 2), round(misclassification_rate, 2)


# Apply confusion matrix metrics
for model_name, y_pred in zip(["VADER", "Naïve Bayes", "XGboost"],
                              [y_pred_vader, y_pred_nb,y_pred_xgb]):
    detected_positive, detected_negative, misclassification_rate = calculate_metrics(y_pred, y_true)


# Convert results to DataFrame
df_misclassification = pd.DataFrame(model_results)

# Display the final misclassification table
print(df_misclassification)
