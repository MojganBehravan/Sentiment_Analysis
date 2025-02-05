# SentimentAnalysis
This project performs sentiment analysis on the Amazon Fine Food Reviews dataset using different models:

Rule-Based Model (VADER)
Traditional Machine Learning (Naïve Bayes)
Deep Learning Models (LSTM with Attention & BERT)
The pipeline allows separate execution of each script, enabling modular testing of each method.

📥 Dataset
The dataset used in this project is available on Kaggle:
🔗 [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

📌 Steps to Download:
Go to the dataset link above.
Download Reviews.csv.
Create a "data" folder in project and place dataset inside the data/ folder.

⚙️ Installation
Make sure you have Python 3.10+ and install required dependencies:
pip install -r requirements.txt

🚀 How to Run the Models
1️⃣ Load Dataset (First Step)
python scripts/data_loader.py
Loads and prepares the Amazon Fine Food Reviews dataset.
2️⃣ Preprocess the Data (Second Step)
python scripts/preprocessing.py
Cleans text (removes special characters, stopwords)
Saves processed dataset (cleaned_reviews.csv)
3️⃣ Run Any Model Separately - Evaluation included:
For VADER run: python scripts/rule_base.py
For Traditional ML run: python scripts/Handling_Unbalanced_Combine_ML.py
For LSTM and BERT run: python scripts/deep_learning.py
I Highly recommended for running LSTM with Attention (Deep Learning) and BERT use kaggle notebook or google colab
📌 Run this in Kaggle or Colab for better performance
🔗 [Amazon Fine Food Reviews (LSTM Model)](https://www.kaggle.com/code/mojganb/amazon-fine-food-reviews)
🔗 [Fine Food Sentiment Analysis (BERT Model)](https://www.kaggle.com/code/mojganb/fine-food-deep-learning)

4️⃣ Visualize Model Performance (Pipeline)
python pipeline.py


