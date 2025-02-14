import pandas as pd

# Load the dataset
df = pd.read_csv("sample_reviews.csv")

# Define sarcasm-related keywords
sarcasm_keywords = [
    "great... not", "love waiting", "best worst", "fantastic service... said no one",
    "yeah right", "just perfect...not", "oh joy", "wonderful... not", "so amazing... not",
    "exactly what I needed... not", "best decision ever... not", "what a treat... not"
]

# Filter reviews containing sarcastic phrases
df_sarcasm = df[df["Text"].str.contains('|'.join(sarcasm_keywords), case=False, na=False)]

# Detect sentiment inconsistency:
positive_words = ["amazing", "great", "excellent", "fantastic", "love", "best"]
negative_words = ["terrible", "awful", "worst", "horrible", "hate", "bad"]

# High-score reviews (4-5) containing negative words
high_score_negative = df[(df["Score"] >= 4) & (df["Text"].str.contains('|'.join(negative_words), case=False, na=False))]

# Low-score reviews (1-2) containing positive words
low_score_positive = df[(df["Score"] <= 2) & (df["Text"].str.contains('|'.join(positive_words), case=False, na=False))]

# Combine all sarcastic reviews and remove duplicates
df_sarcastic_reviews = pd.concat([df_sarcasm, high_score_negative, low_score_positive]).drop_duplicates()
#We use pd.concat([...]) to combine three different filtering approaches into one dataframe.
#Since some reviews may be identified by multiple filters, we remove duplicates to avoid redundancy.

# Display sarcastic reviews with IDs
print(df_sarcastic_reviews[['Id', 'Text', 'Score']])
