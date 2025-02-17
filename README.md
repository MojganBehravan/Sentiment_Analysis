# Sentiment Analysis

This project performs sentiment analysis on the **Amazon Fine Food Reviews** dataset using different models:

- **Rule-Based Model (VADER)**
- **Traditional Machine Learning (Naïve Bayes)**
- **Deep Learning Models (LSTM with Attention & BERT)**

The pipeline allows separate execution of each script, enabling modular testing of each method.

---

## 📥 Dataset

The dataset used in this project is available on **Kaggle**:

🔗 [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### 📌 Steps to Download:
1. Go to the dataset link above.
2. Download **`Reviews.csv`**.
3. Create a `"data"` folder in the project and place the dataset inside the `data/` folder.

---

## ⚙️ Installation

Make sure you have **Python 3.10+** and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Models

### 1️⃣ Load Dataset (First Step)
```bash
python scripts/data_loader.py
```
- Loads and prepares the **Amazon Fine Food Reviews dataset**.

### 2️⃣ Preprocess the Data (Second Step)
```bash
python scripts/preprocessing.py
```
- Cleans text (removes special characters, stopwords).
- Saves processed dataset as **`cleaned_reviews.csv`**.

---

## 📊 Run Any Model Separately - Evaluation Included:

- **For VADER**, run:
  ```bash
  python scripts/rule_base.py
  ```

- **For Naïve Bayes (Machine Learning)**
  ```bash
  python scripts/Handling_Unbalanced_Combine_ML.py
  ```

- **For XGBoost**, use the **online version**:  
  🔗 [XGBoost Model on Kaggle](https://www.kaggle.com/code/mojganb/xgboost)

- **For LSTM and BERT**, run:
  ```bash
  python scripts/deep_learning.py
  ```

### 🔹 **Recommended for LSTM & BERT**
**It is highly recommended to run LSTM with Attention (Deep Learning) and BERT on Kaggle Notebook or Google Colab** for better performance.

📌 **Run these models in Kaggle or Colab for better results:**

🔗 [Amazon Fine Food Reviews (LSTM Model)](https://www.kaggle.com/code/mojganb/amazon-fine-food-lstm)  
🔗 [Fine Food Sentiment Analysis (BERT Model)](https://www.kaggle.com/code/mojganb/amazon-fine-food-bert)

---
