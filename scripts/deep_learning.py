import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data,preprocess_lstm

# === Step 1: Load Dataset ===
file_path = '../data/cleaned_reviews.csv'
df = pd.read_csv(file_path)

# Map sentiment labels to numbers
label_mapping = {'positive': 2, 'negative': 0, 'neutral': 1}
df['sentiment_label'] = df['Sentiment'].map(label_mapping)
df = preprocess_data(df)

print(df.columns)

# === Step 2: Tokenize and Convert Text to Sequences ===
max_words = 5000
max_seq_length = 100

X_train, X_test, Y_train, Y_test = train_test_split(df['Cleaned_Text'], df['sentiment_label'], test_size=0.2, random_state=42)

X_train_df = pd.DataFrame(X_train, columns=['Cleaned_Text'])
X_test_df = pd.DataFrame(X_test, columns=['Cleaned_Text'])

#  Use the correct input format for LSTM
X_train_padded, X_test_padded, tokenizer = preprocess_lstm(X_train_df, X_test_df, max_words, max_seq_length)

# === Step 3: Define LSTM Model ===
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),  #  Removed deprecated `input_length`
    LSTM(64, return_sequences=True),
    Dropout(0.2),  #Helps model generalize sarcasm better
    LSTM(32),
    Dense(3, activation='softmax')  # 3 output classes: positive, neutral, negative
])


# === Step 4: Compile Model ===
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
# === Step 5: Train Model (Fix Applied Here) ===
model.fit(X_train_padded, Y_train, validation_split=0.2, epochs=5, batch_size=64)
end_time = time.time()
training_time = end_time - start_time  # Total time taken

start_inference_time = time.time()
loss, accuracy = model.evaluate(X_test_padded, Y_test)
end_inference_time = time.time()

inference_time = end_inference_time - start_inference_time

# Print Computation Time Results
print(f"\n Training Time: {training_time:.2f} seconds")
print(f" Inference Time: {inference_time:.2f} seconds")
print(f" Test Accuracy: {accuracy:.4f}")

# === Step 6: Save Tokenizer (for inference later) ===


print(" Model training complete! ")


# === ===========================  BERT ============================================
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
from sklearn.metrics import classification_report
from scripts.preprocessing import  preprocess_and_split,preprocess_bert

# === Step 1: Load Dataset and Preprocess ===
file_path = '/kaggle/working/cleaned_reviews.csv'  # Your cleaned dataset
df = pd.read_csv(file_path)
df=preprocess_data(df)
# Call function to split dataset and save to CSV
output_files = preprocess_and_split(df)  #  Calls existing function

# Retrieve train and test file paths
train_file = output_files['train_file']
test_file = output_files['test_file']

# Load preprocessed train and test data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_data['Text'] = train_data['Text'].astype(str).fillna("")
test_data['Text'] = test_data['Text'].astype(str).fillna("")

assert 'Text' in train_data.columns, "Error: 'Text' column is missing in train_data!"
assert 'Text' in test_data.columns, "Error: 'Text' column is missing in test_data!"

print("Sample Train Text:", train_data['Text'].head())
print("Sample Test Text:", test_data['Text'].head())

print(f"train columns: {train_data.columns}")  # Check available columns
print(f"test columns: {test_data.columns}" )

# === Step 2: Load BERT Tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize train and test data using the existing function
train_tokens = preprocess_bert(train_data, tokenizer)
test_tokens = preprocess_bert(test_data, tokenizer)

# === Step 3: Extract Labels and Convert to Tensor ===
train_labels = tf.convert_to_tensor(train_data['Label'].values, dtype=tf.int32)
test_labels = tf.convert_to_tensor(test_data['Label'].values, dtype=tf.int32)

# === Step 4: Load BERT Model ===
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define optimizer
optimizer = AdamWeightDecay(learning_rate=5e-5, weight_decay_rate=0.01)

# Compile the model (DO NOT pass `loss=model.compute_loss`)
model.compile(optimizer=optimizer, metrics=['accuracy'])

# === Step 5: Train the Model ===
model.fit(
    x={'input_ids': train_tokens['input_ids'], 'attention_mask': train_tokens['attention_mask']},
    y=train_labels,
    validation_data=(
        {'input_ids': test_tokens['input_ids'], 'attention_mask': test_tokens['attention_mask']},
        test_labels
    ),
    epochs=3,
    batch_size=16
)

# === Step 6: Make Predictions ===
y_pred_logits = model.predict({'input_ids': test_tokens['input_ids'], 'attention_mask': test_tokens['attention_mask']}).logits
y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()  # Convert to NumPy array

# === Step 7: Evaluate the Model ===
print("\nBERT Sentiment Analysis Results:")
print(classification_report(test_data['Label'], y_pred_labels))

# === Step 8: Save Predictions to CSV ===
test_data['Predicted_Sentiment'] = y_pred_labels
test_data.to_csv('/kaggle/working/bert_results.csv', index=False)
print("\nBERT predictions saved to '/kaggle/working/bert_results.csv'.")
