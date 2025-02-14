
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
from scripts.preprocessing import preprocess_naive_bayes_without_pca
import pandas as pd
import time
import joblib

train_file='../data/train_data.csv'
test_file='../data/test_data.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data['Text']
Y_train = train_data['Label']
X_test = test_data['Text']
Y_test = test_data['Label']

X_train = X_train.fillna("")
X_test = X_test.fillna("")
start_train_time = time.perf_counter() # Start training timer
X_train_nb, X_test_nb, tfidf = preprocess_naive_bayes_without_pca(X_train, X_test)

undersampler = RandomUnderSampler( random_state=42)
X_resampled, Y_resampled = undersampler.fit_resample(X_train_nb, Y_train)

smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_nb, Y_train)


# Train Naïve Bayes


nb_model = MultinomialNB()
nb_model.fit(X_train_balanced, Y_train_balanced)
end_train_time = time.perf_counter()  # End training timer
train_time = end_train_time - start_train_time

# Evaluate the model on the test set
start_test_time = time.perf_counter()  # Start testing timer
Y_test_pred = nb_model.predict(X_test_nb)
end_test_time = time.perf_counter()  # End testing timer

# Save trained model
joblib.dump(nb_model, "../data/naive_bayes_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "../data/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# Calculate testing time
testing_time = end_test_time - start_test_time
print(f"Prediction Time: {testing_time:.2f} seconds")
print(f"Train Time: {train_time:.2f} seconds")

print("Test Accuracy (Combine SMOTE with random sampling):", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (SCombine SMOTE with random sampling):")
print(classification_report(Y_test, Y_test_pred))