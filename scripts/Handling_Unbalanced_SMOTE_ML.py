from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
from scripts.preprocessing import preprocess_naive_bayes_without_pca
from sklearn.linear_model import LogisticRegression

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
smote = SMOTE(sampling_strategy="not majority", random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_nb, Y_train)

# Display new class distribution
print("Class distribution after SMOTE:")
print(pd.Series(Y_train_balanced).value_counts())

# Train Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_balanced, Y_train_balanced)


# Evaluate the model on the test set
Y_test_pred = nb_model.predict(X_test_nb)
print("Test Accuracy (SMOTE only):", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (SMOTE only):")
print(classification_report(Y_test, Y_test_pred))

# with under sampling
undersampler = RandomUnderSampler(random_state=42)
X_train_balanced, Y_train_balanced = undersampler.fit_resample(X_train_nb, Y_train)


# Train Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_balanced, Y_train_balanced)


# Evaluate the model on the test set
Y_test_pred = nb_model.predict(X_test_nb)
print("Test Accuracy (SMOTE under sampling):", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (SMOTE under sampling):")
print(classification_report(Y_test, Y_test_pred))

### using logistic regression
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_nb, Y_train)

print("Class distribution after SMOTE:")
print(pd.Series(Y_train_balanced).value_counts())

# Train Logistic Regression with class weights
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
log_reg.fit(X_train_balanced, Y_train_balanced)

# Predict on test data
y_probs = log_reg.predict_proba(X_test_nb)

# Apply threshold tuning for class 1
y_pred = []
for prob in y_probs:
    if prob[1] > 0.6:  # Custom threshold for class 1
        y_pred.append(1)
    else:
        y_pred.append(prob.argmax())  # Default to the class with the highest probability


# Evaluate the model
test_accuracy = accuracy_score(Y_test, y_pred)
print("\nTest Accuracy (Weighted Logistic Regression with SMOTE):", test_accuracy)
print("\nClassification Report (Weighted Logistic Regression with SMOTE):")
print(classification_report(Y_test, y_pred))