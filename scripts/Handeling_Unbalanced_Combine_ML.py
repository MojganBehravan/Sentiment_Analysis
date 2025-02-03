
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
from scripts.preprocessing import preprocess_naive_bayes_without_pca
import pandas as pd

train_file='../data/train_data.csv'
test_file='../data/test_data.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data['Text']
Y_train = train_data['Label']
X_test = test_data['Text']
Y_test = test_data['Label']

X_train = X_train.fillna("")
X_test = X_test.fillna("")

X_train_nb, X_test_nb, tfidf = preprocess_naive_bayes_without_pca(X_train, X_test)

undersampler = RandomUnderSampler(random_state=42)
X_resampled, Y_resampled = undersampler.fit_resample(X_train_nb, Y_train)

smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_nb, Y_train)


# Train Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_balanced, Y_train_balanced)


# Evaluate the model on the test set
Y_test_pred = nb_model.predict(X_test_nb)
print("Test Accuracy (Combine SMOTE with Undersampling):", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (SCombine SMOTE with Undersampling):")
print(classification_report(Y_test, Y_test_pred))