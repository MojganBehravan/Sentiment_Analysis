from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scripts.preprocessing import preprocess_naive_bayes_without_pca
from sklearn.naive_bayes import MultinomialNB

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
print('end of preprocessing TF-IDF')
# === Baseline Model (Without Any Resampling) ===
nb_model = MultinomialNB()
nb_model.fit(X_train_nb, Y_train)
Y_test_pred_baseline = nb_model.predict(X_test_nb)

print("\n=== Baseline Model (No Resampling) ===")
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred_baseline))
print(classification_report(Y_test, Y_test_pred_baseline))

# === SMOTE Oversampling ===
smote = SMOTE(sampling_strategy={0: 40339, 1: 40339, 2: 40339}, random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train_nb, Y_train)

nb_model.fit(X_train_smote, Y_train_smote)
Y_test_pred_smote = nb_model.predict(X_test_nb)

print("\n=== Model with SMOTE Oversampling ===")
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred_smote))
print(classification_report(Y_test, Y_test_pred_smote))

# === Undersampling Majority Class ===
undersampler = RandomUnderSampler(sampling_strategy={0: 1039, 1: 1039, 2: 1039} , random_state=42)
X_train_under, Y_train_under = undersampler.fit_resample(X_train_nb, Y_train)

nb_model.fit(X_train_under, Y_train_under)
Y_test_pred_under = nb_model.predict(X_test_nb)

print("\n=== Model with Undersampling ===")
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred_under))
print(classification_report(Y_test, Y_test_pred_under))

# === Hybrid Approach (SMOTE + Undersampling) ===
X_train_hybrid, Y_train_hybrid = smote.fit_resample(X_train_nb, Y_train)
X_train_hybrid, Y_train_hybrid = undersampler.fit_resample(X_train_hybrid, Y_train_hybrid)

nb_model.fit(X_train_hybrid, Y_train_hybrid)
Y_test_pred_hybrid = nb_model.predict(X_test_nb)

print("\n=== Model with SMOTE + Undersampling ===")
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred_hybrid))
print(classification_report(Y_test, Y_test_pred_hybrid))

import matplotlib.pyplot as plt
import numpy as np

# Accuracy Scores
methods = ['No Resampling', 'SMOTE Oversampling', 'Undersampling', 'Oversampling + Undersampling']
accuracy = [0.813, 0.739, 0.642, 0.648]

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(methods, accuracy, color=['blue', 'orange', 'red', 'green'])
plt.xlabel("Resampling Technique")
plt.ylabel("Accuracy")
plt.title("Performance Comparison of Resampling Techniques")
plt.ylim(0.6, 0.85)

# Show value labels on bars
for i, acc in enumerate(accuracy):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center', fontsize=12)

plt.show()
