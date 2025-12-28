import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# LOAD DATASET WITH COLUMN NAMES
# ===============================
column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(
    r"C:\Users\st\Desktop\Python\first project\diabetes.csv", names=column_names, header=0)

print("Dataset loaded")
print(df.head())

# ===============================
# DATA CLEANING
# ===============================
cols_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# MODELS
# ===============================
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
log_acc = accuracy_score(y_test, log_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Logistic Regression Accuracy: {log_acc:.3f}")
print(f"Random Forest Accuracy: {rf_acc:.3f}")

if rf_acc > log_acc:
    print("✅ Best Model: Random Forest")
else:
    print("✅ Best Model: Logistic Regression")

import joblib

joblib.dump(log_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")
