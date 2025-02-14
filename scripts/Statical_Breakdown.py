# Statistical breakdown of sentiment classes
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = '../data/Reviews.csv'
df = pd.read_csv(file_path)

# Map sentiment labels (optional)
label_mapping = {'positive': 2, 'negative': 0, 'neutral': 1}
df['Sentiment'] = df['Score'].apply(lambda x: 2 if x >= 4 else (1 if x == 3 else 0))

# Calculate distribution
sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
sentiment_counts = sentiment_counts.rename({2: "Positive", 1: "Neutral", 0: "Negative"})

# Display breakdown
print("\nSentiment Distribution:")
print(sentiment_counts)

# Optional: Visualization
# Plot the chart
plt.figure(figsize=(8, 5))
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'blue', 'red'])

# Add percentages on top of each bar
for bar, value in zip(bars, sentiment_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Position at the center of the bar
        bar.get_height() + 1,              # Slightly above the bar
        f"{value:.1f}%",                   # Format as percentage
        ha='center',                       # Horizontal alignment
        fontsize=12                        # Font size
    )

# Set chart title and labels
plt.title("Percentage Distribution of Sentiments in Amazon Fine Food Reviews", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Count the occurrences of each score (1-5)
score_counts = df['Score'].value_counts(normalize=True) * 100  # Percentage distribution
score_counts = score_counts.sort_index()  # Ensure scores are in ascending order

# Display the score distribution
print("\nScore Distribution (Percentage):")
print(score_counts)

# Visualization
plt.figure(figsize=(8, 5))
bars = plt.bar(score_counts.index, score_counts.values, color=['red', 'orange', 'yellow', 'blue', 'green'])
# Add percentages on top of each bar
for bar, value in zip(bars, score_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Position at the center of the bar
        bar.get_height() + 1,              # Slightly above the bar
        f"{value:.1f}%",                   # Format as percentage
        ha='center',                       # Horizontal alignment
        fontsize=12                        # Font size
    )
plt.title("Percentage Distribution of Scores (1-5)", fontsize=14)
plt.xlabel("Score", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
