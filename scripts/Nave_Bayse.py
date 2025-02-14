import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessing import preprocess_naive_bayes_without_pca
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load Data
train_file = '../data/train_data.csv'
test_file = '../data/test_data.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data['Text']
Y_train = train_data['Label']
X_test = test_data['Text']
Y_test = test_data['Label']

X_train = X_train.fillna("")
X_test = X_test.fillna("")

# Preprocess Text (TF-IDF)
X_train_nb, X_test_nb, tfidf = preprocess_naive_bayes_without_pca(X_train, X_test)
def train_evaluate_model(X_train_resampled, Y_train_resampled, X_test, Y_test, model_name=""):
    """Train and evaluate a Naïve Bayes model"""
    model = MultinomialNB()
    model.fit(X_train_resampled, Y_train_resampled)
    Y_test_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_test_pred)
    print(f"\n=== {model_name} ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(Y_test, Y_test_pred))
    return accuracy
print('end of preprocessing TF-IDF')
# === Baseline Model (Without Any Resampling) ===
baseline_acc = train_evaluate_model(X_train_nb, Y_train, X_test_nb, Y_test, "Baseline Model (No Resampling)")

# ===  Apply SMOTE (Oversampling) ===
smote = SMOTE(sampling_strategy="not majority", random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train_nb, Y_train)
smote_acc = train_evaluate_model(X_train_smote, Y_train_smote, X_test_nb, Y_test, "SMOTE Oversampling")

# ===  Apply Random Undersampling ===
undersampler = RandomUnderSampler(sampling_strategy={0: 1039, 1: 1039, 2: 3000}, random_state=42)
X_train_under, Y_train_under = undersampler.fit_resample(X_train_nb, Y_train)
undersampling_acc = train_evaluate_model(X_train_under, Y_train_under, X_test_nb, Y_test, "Random Undersampling")


# ===  SMOTE + Undersampling (Hybrid) ===
X_train_smote_under, Y_train_smote_under = smote.fit_resample(X_train_under, Y_train_under)
smote_under_acc = train_evaluate_model(X_train_smote_under, Y_train_smote_under, X_test_nb, Y_test, "SMOTE + Undersampling")
# ===  Class Distribution Visualization ===
def plot_class_distribution(before_counts, after_counts, title):
    df_counts = pd.DataFrame({
        'Sentiment': ['Negative', 'Neutral', 'Positive'] * 2,
        'Count': np.concatenate([before_counts, after_counts]),
        'Dataset': ['Before Resampling'] * 3 + ['After Resampling'] * 3
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_counts, x='Sentiment', y='Count', hue='Dataset')
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.xlabel("Sentiment Class")
    plt.legend(title="Dataset Type")
    plt.show()

# Show class distribution before & after SMOTE
before_counts = np.bincount(Y_train)
after_counts = np.bincount(Y_train_smote)
plot_class_distribution(before_counts, after_counts, "Class Distribution Before & After SMOTE")

# Show class distribution before & after undersampling
before_counts = np.bincount(Y_train)
after_counts = np.bincount(Y_train_under)
plot_class_distribution(before_counts, after_counts, "Class Distribution Before & After Undersampling")

# === Step 9: Accuracy Comparison Bar Chart ===
models = ["Baseline", "SMOTE", "Undersampling", "SMOTE + Undersampling"]
accuracies = [baseline_acc, smote_acc, undersampling_acc, smote_under_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['red', 'orange', 'yellow', 'blue', 'green'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
             f"{height * 100:.2f}%", ha='center', fontsize=12)
plt.title("Comparison of Accuracy Across Resampling Methods")
plt.xlabel("Resampling Method")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 1.05)
plt.show()

print("Resampling experiments completed successfully!")
