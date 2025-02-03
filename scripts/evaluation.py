import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_model(y_true, y_pred, model_name, start_time, end_time):
    """
    Evaluates model performance and prints accuracy, precision, recall, F1-score, and computation time.
    Also displays a confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\n {model_name} Performance:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f"⚖  F1-score: {f1:.4f}")
    print(f" Computation Time: {end_time - start_time:.2f} seconds")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def compare_models(results):
    """
    Compare all models in a summary table.
    :param results: List of dictionaries containing model results.
    """
    df_results = pd.DataFrame(results)
    print("\n Model Performance Summary:")
    print(df_results)

    # Plot comparison
    plt.figure(figsize=(10,5))
    sns.barplot(x="Model", y="Accuracy", data=df_results, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()
