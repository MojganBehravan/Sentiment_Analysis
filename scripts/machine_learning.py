import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scripts.preprocessing import preprocess_data,preprocess_naive_bayes_without_pca, preprocess_and_split,preprocess_naive_bayes_with_pca


# Load the cleaned dataset
file_path = '../data/cleaned_reviews.csv'
data = pd.read_csv(file_path)

# Preprocess data
data = preprocess_data(data)
output_files = preprocess_and_split(data)

# === Step 3: Load Preprocessed Train/Test Data ===
train_file = output_files['train_file']
test_file = output_files['test_file']

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Extract features and labels
X_train, Y_train = train_data['Text'], train_data['Label']
X_test, Y_test = test_data['Text'], test_data['Label']

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# Preprocess data for Naïve Bayes without PCA
X_train_nb, X_test_nb, tfidf = preprocess_naive_bayes_without_pca(X_train, X_test)
print('End of Naïve Bayes preprocessing')

# Train and evaluate Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_nb, Y_train)

# Predict on test data
Y_test_pred = nb_model.predict(X_test_nb)

# Evaluate on training set
Y_train_pred = nb_model.predict(X_train_nb)
train_accuracy = accuracy_score(Y_train, Y_train_pred)
print("Training Accuracy:", train_accuracy)

# Evaluate model
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Test Accuracy:", test_accuracy)

# Classification report
print("\nNaïve Bayes Sentiment Analysis Results:")
print(classification_report(Y_test, Y_test_pred))

# ---- Case 2: With PCA ----
print("\nPreprocessing for Naïve Bayes with PCA...")
X_train_pca, X_test_pca, tfidf_pca, pca = preprocess_naive_bayes_with_pca(X_train, X_test, max_features=2000, n_components=300)

 #Train and evaluate Naïve Bayes model (with PCA)
nb_model_with_pca = MultinomialNB()
nb_model_with_pca.fit(X_train_pca, Y_train)

# Predict on test data
Y_test_pred_pca = nb_model_with_pca.predict(X_test_pca)

# Evaluate the model (with PCA)
test_accuracy_pca = accuracy_score(Y_test, Y_test_pred_pca)
print("\nNaïve Bayes with PCA Results:")
print(f"Test Accuracy: {test_accuracy_pca}")
print(classification_report(Y_test, Y_test_pred_pca))

# performance without PCA is already significantly better than with PCA, so Choose without PCA with some changes
