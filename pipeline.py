from scripts.evaluation import compare_models

results = [
    {"Model": "VADER (Rule-Based)", "Accuracy": 0.78, "Precision": 0.72, "Recall": 0.78, "F1-score": 0.74},
    {"Model": "Naïve Bayes (TF-IDF)", "Accuracy": 0.74, "Precision": 0.83, "Recall": 0.74, "F1-score": 0.77},
    {"Model": "LSTM with Attention", "Accuracy": 0.90, "Precision": 0.89, "Recall": 0.90, "F1-score": 0.89},
    {"Model": "BERT Transformer", "Accuracy": 0.91, "Precision": 0.90, "Recall": 0.91, "F1-score": 0.91}
]

compare_models(results)