import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
DATA_PATH = '../data/Reviews.csv'  # Adjust path as needed
df = pd.read_csv(DATA_PATH)

# Basic exploration
print("Dataset Shape:", df.shape)
print("\nColumns in the dataset:", df.columns)
print("\nSample Data:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Drop rows with missing values (if necessary)
df = df.dropna(subset=['Text', 'Score'])
print("\nShape after dropping missing values:", df.shape)

# Distribution of 'Score' column
print("\nScore Distribution:")
print(df['Score'].value_counts())

# Visualize the score distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Score', palette='viridis')
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()

# Simplify sentiment labels
def assign_sentiment(score):
    if score in [4, 5]:
        return 'positive'
    elif score == 3:
        return 'neutral'
    elif score in [1, 2]:
        return 'negative'

df['Sentiment'] = df['Score'].apply(assign_sentiment)

# Verify sentiment distribution
print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sentiment', palette='viridis', order=['positive', 'neutral', 'negative'])
plt.title("Distribution of Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Save cleaned data for further processing
df.to_csv('../data/cleaned_reviews.csv', index=False)
print('the cleand data saved to directory')