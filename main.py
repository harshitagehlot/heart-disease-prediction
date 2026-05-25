# ============================================
# HEART DISEASE PREDICTION PROJECT
# ============================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/heart.csv")

print("\n========================")
print("FIRST 5 ROWS")
print("========================")
print(df.head())

print("\n========================")
print("DATASET INFO")
print("========================")
print(df.info())

print("\n========================")
print("MISSING VALUES")
print("========================")
print(df.isnull().sum())

# =========================
# HANDLE MISSING VALUES
# =========================

df.fillna(df.mean(numeric_only=True), inplace=True)

# =========================
# DATA VISUALIZATION
# =========================

# Target Distribution

sns.countplot(x='target', data=df)

plt.title("Heart Disease Distribution")

plt.show()

# Correlation Heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Feature Correlation Heatmap")

plt.show()

# =========================
# FEATURES & TARGET
# =========================

X = df.drop('target', axis=1)

y = df['target']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================
# LOGISTIC REGRESSION
# =========================

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# =========================
# SVM MODEL
# =========================

svm = SVC(probability=True)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# =========================
# RANDOM FOREST
# =========================

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# =========================
# XGBOOST
# =========================

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_model(y_test, predictions, model_name):

    print("\n========================")
    print(model_name)
    print("========================")

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# =========================
# EVALUATE ALL MODELS
# =========================

evaluate_model(y_test, lr_pred, "Logistic Regression")

evaluate_model(y_test, svm_pred, "SVM")

evaluate_model(y_test, rf_pred, "Random Forest")

evaluate_model(y_test, xgb_pred, "XGBoost")

# =========================
# MODEL COMPARISON GRAPH
# =========================

models = ['LR', 'SVM', 'RF', 'XGB']

scores = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, xgb_pred)
]

plt.figure(figsize=(8,5))

plt.bar(models, scores)

plt.ylabel("Accuracy")

plt.title("Model Comparison")

plt.show()

# =========================
# FEATURE IMPORTANCE GRAPH
# =========================

importance = rf.feature_importances_

plt.figure(figsize=(10,6))

plt.barh(X.columns, importance)

plt.xlabel("Importance")

plt.title("Feature Importance")

plt.show()

# =========================
# SAVE BEST MODEL
# =========================

joblib.dump(rf, "models/heart_model.pkl")

print("\nModel saved successfully!")

# =========================
# SAMPLE PREDICTIONS
# =========================

sample_predictions = rf.predict(X_test[:10])

print("\n========================")
print("SAMPLE PREDICTIONS")
print("========================")

for i, pred in enumerate(sample_predictions):

    if pred == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    print(f"Patient {i+1}: {result}")

# =========================
# SAVE PREDICTIONS TO TXT
# =========================

with open("prediction_output.txt", "w") as file:

    file.write("Heart Disease Predictions\n\n")

    for i, pred in enumerate(sample_predictions):

        if pred == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        file.write(f"Patient {i+1}: {result}\n")

print("\nPredictions saved in prediction_output.txt")

# =========================
# PREDICTION RESULT TABLE
# =========================

results = pd.DataFrame({
    "Actual": y_test[:10].values,
    "Predicted": sample_predictions
})

print("\n========================")
print("PREDICTION RESULTS TABLE")
print("========================")

print(results)

# =========================
# VISUALIZE PREDICTIONS
# =========================

labels = []

for pred in sample_predictions:

    if pred == 1:
        labels.append("Disease")
    else:
        labels.append("No Disease")

plt.figure(figsize=(10,5))

plt.bar(range(len(labels)), sample_predictions)

plt.xticks(range(len(labels)), labels)

plt.ylabel("Prediction")

plt.title("Sample Heart Disease Predictions")

plt.show()

# ============================================
# END OF PROJECT
# ============================================