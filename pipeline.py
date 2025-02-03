from scripts.evaluation import compare_models

results = [
    {"Model": "VADER (Rule-Based)", "Accuracy": 0.72, "Precision": 0.68, "Recall": 0.65, "F1-score": 0.66},
    {"Model": "Naïve Bayes (TF-IDF)", "Accuracy": 0.80, "Precision": 0.79, "Recall": 0.78, "F1-score": 0.78},
    {"Model": "LSTM with Attention", "Accuracy": 0.85, "Precision": 0.84, "Recall": 0.83, "F1-score": 0.84},
    {"Model": "BERT Transformer", "Accuracy": 0.91, "Precision": 0.90, "Recall": 0.89, "F1-score": 0.90}
]

compare_models(results)