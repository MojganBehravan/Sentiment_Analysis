import streamlit as st
import joblib
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Load models and vectorizers
nb_model = joblib.load('data/naive_bayes_model.pkl')
tfidf = joblib.load('data/tfidf_vectorizer.pkl')

# Initialize VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load LSTM model and tokenizer
lstm_model = tf.keras.models.load_model('data/lstm_model-Resampling.h5')
lstm_tokenizer = joblib.load('data/tokenizer-Resampling.pkl')

# Load BERT sentiment analysis pipeline
bert_tokenizer= AutoTokenizer.from_pretrained('distillbert_efficiencey_tokenizer')
# Load model
bert_model = TFAutoModelForSequenceClassification.from_pretrained('distillbert_efficiencey_model')


# Streamlit UI
#st.title("Real-Time Sentiment Analysis")
#st.subheader("Enter text to analyze sentiment using different models")

st.markdown("<h2 style='text-align: center;'>Real-Time Sentiment Analysis</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter text to analyze sentiment using different models</h4>", unsafe_allow_html=True)

# User input
user_input = st.text_area("", "", placeholder="Type your sentence here...")

if st.button("Analyze Sentiment"):
    if user_input:
        st.write("### Results from Different Models:")

        # Naïve Bayes Model
        text_vectorized = tfidf.transform([user_input])
        nb_prediction = nb_model.predict(text_vectorized)
        nb_sentiment = "Positive 😊" if nb_prediction[0] == 1 else "Negative 😞"
        st.write(f"**Naïve Bayes:** {nb_sentiment}")

        # VADER Model
        vader_scores = sia.polarity_scores(user_input)
        vader_sentiment = "Positive 😊" if vader_scores['compound'] >= 0 else "Negative 😞"
        st.write(f"**VADER:** {vader_sentiment} (Score: {vader_scores['compound']:.2f})")

        # LSTM Model
        lstm_sequence = lstm_tokenizer.texts_to_sequences([user_input])
        lstm_padded = tf.keras.preprocessing.sequence.pad_sequences(lstm_sequence, maxlen=100)
        lstm_prediction = lstm_model.predict(lstm_padded)[0][0]
        lstm_sentiment = "Positive 😊" if lstm_prediction >= 0.5 else "Negative 😞"
        st.write(f"**LSTM:** {lstm_sentiment} (Score: {lstm_prediction:.2f})")

        # BERT Model
        def bert_predict(text):
            inputs = bert_tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
            outputs = bert_model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1)
            sentiment_label = tf.argmax(predictions, axis=-1).numpy()[0]
            sentiment = "Positive 😊" if sentiment_label == 2 else "Negative 😞"
            return sentiment

        bert_sentiment = bert_predict(user_input)
        st.write(f"**BERT:** {bert_sentiment}")

    else:
        st.warning("Please enter text to analyze.")

        # streamlit run app.py