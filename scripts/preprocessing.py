import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Download NLTK resources
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('punkt')

# Define shared text cleaning function
def clean_text(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = nltk.word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Define stopwords
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)


# Define function to remove illegal characters
def remove_illegal_characters(text):
    if isinstance(text, str):
        # Remove all non-printable characters
        return re.sub(r'[^\x20-\x7E\u00A0-\u00FF]', '', text)  # Added more Unicode range
    return text


# Preprocessing pipeline for all models
def preprocess_data(df):
    df['Text'] = df['Text'].fillna("")
    df['Cleaned_Text'] = df['Text'].apply(clean_text)  # Apply text cleaning
    df['Cleaned_Text'] = df['Cleaned_Text'].apply(remove_illegal_characters)
    return df


def preprocess_naive_bayes_without_pca(X_train, X_test, max_features=5000):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Return the TF-IDF matrix and the vectorizer
    return X_train_tfidf, X_test_tfidf, tfidf


# Specific preprocessing for Naïve Bayes
def preprocess_naive_bayes_with_pca(X_train, X_test, max_features=2000, n_components=300):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_reduced = pca.fit_transform(X_train_tfidf.toarray())  # PCA requires dense matrices
    X_test_reduced = pca.transform(X_test_tfidf.toarray())

    # Shift to non-negative values for MultinomialNB
    X_train_reduced -= X_train_reduced.min(axis=0)
    X_test_reduced -= X_test_reduced.min(axis=0)

    # Return PCA-transformed data, TF-IDF vectorizer, and PCA object
    return X_train_reduced, X_test_reduced, tfidf, pca


# Specific preprocessing for LSTM
def preprocess_lstm(train_data, test_data, max_words=5000, max_seq_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data['Cleaned_Text'])

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_data['Cleaned_Text']), maxlen=max_seq_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_data['Cleaned_Text']), maxlen=max_seq_len)

    return X_train_seq, X_test_seq, tokenizer


# Specific preprocessing for BERT
def preprocess_bert(data, tokenizer, max_len=128):
    tokens = tokenizer(
        list(data['Cleaned_Text']),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )
    return tokens


def preprocess_and_split(df, label_column='Sentiment', text_column='Cleaned_Text', test_size=0.2, random_state=42):
    # Map sentiment labels to numeric values
    label_mapping = {'positive': 2, 'negative': 0, 'neutral': 1}
    df['sentiment_label'] = df[label_column].map(label_mapping)

    # Features and Labels
    X = df[text_column]  # Features (text)
    Y = df['sentiment_label']  # Numeric labels

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Replace NaN values with empty strings
    X_train = X_train.fillna("")
    X_test = X_test.fillna("")

    # Save training and testing sets to CSV in Kaggle's working directory
    train_data = pd.DataFrame({'Text': X_train, 'Label': Y_train})
    test_data = pd.DataFrame({'Text': X_test, 'Label': Y_test})

    train_file_path = '../data/train_data.csv'
    test_file_path = '../data/test_data.csv'

    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training data saved to: {train_file_path}")
    print(f"Testing data saved to: {test_file_path}")

    return {
        'train_file': train_file_path,
        'test_file': test_file_path
    }