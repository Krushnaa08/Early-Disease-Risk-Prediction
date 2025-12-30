import pandas as pd
import numpy as np
import os

os.makedirs("models", exist_ok=True)

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------- LOAD DATA ----------------
df = pd.read_csv(
    r"C:\Users\st\Desktop\Python\first project\diabetes.csv",

)
print(df.columns.tolist())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------- LOGISTIC REGRESSION ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy: {lr_acc:.3f}")


# ---------------- KNN MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, y_pred_knn)

print(f"KNN Accuracy (k=5): {knn_acc:.3f}")


# ---------------- RANDOM FOREST ----------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {rf_acc:.3f}")


# ---------------- SAVE BEST MODEL ----------------
# We deploy Logistic Regression for interpretability

joblib.dump(lr, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Best model and scaler saved successfully.")
