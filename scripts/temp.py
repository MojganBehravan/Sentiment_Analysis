from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/path/to/your-dataset.csv')  # Update with the actual dataset path
X = df['Cleaned_Text']  # Features
y = df['Sentiment']     # Labels

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---- Case 1: Without PCA ----
nb_model_no_pca = MultinomialNB()
nb_model_no_pca.fit(X_train_tfidf, y_train)

# Evaluate
y_train_pred_no_pca = nb_model_no_pca.predict(X_train_tfidf)
y_test_pred_no_pca = nb_model_no_pca.predict(X_test_tfidf)

print("Without PCA:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_no_pca))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_no_pca))
print(classification_report(y_test, y_test_pred_no_pca))

# ---- Case 2: With PCA ----
pca = PCA(n_components=300, random_state=42)
X_train_pca = pca.fit_transform(X_train_tfidf.toarray())  # Convert sparse matrix to dense
X_test_pca = pca.transform(X_test_tfidf.toarray())

nb_model_pca = MultinomialNB()
nb_model_pca.fit(X_train_pca, y_train)

# Evaluate
y_train_pred_pca = nb_model_pca.predict(X_train_pca)
y_test_pred_pca = nb_model_pca.predict(X_test_pca)

print("\nWith PCA:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_pca))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_pca))
print(classification_report(y_test, y_test_pred_pca))
