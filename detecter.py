# ==============================
# Diabetes Risk Prediction (Non-Clinical)
# Phase 1: Data Loading & Cleaning
# ==============================

import pandas as pd
import numpy as np

# --------- 1. Define column names ----------
columns = [
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

# --------- 2. Load dataset ----------
df = pd.read_csvdf = pd.read_csv(
    r"C:\Users\st\Desktop\Python\first project\diabetes.csv",
     header=None,
    names=columns
)

   

print("\nDataset loaded successfully\n")
print(df.info())

# --------- 3. Replace invalid zeros ----------
invalid_zero_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

# --------- 4. Fill missing values ----------
for col in invalid_zero_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values handled successfully\n")
print(df.isnull().sum())

# --------- 5. Separate features and target ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nData preprocessing completed successfully")

# ==============================
# Phase 2: Model Training
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------- 6. Train-test split ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------- 7. Feature scaling ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------- 8. Train Logistic Regression ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --------- 9. Evaluate ----------
y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Phase 3: Risk Level Prediction
# ==============================

# Get probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

def risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

# Show sample predictions
print("\nSample Risk Predictions:\n")

for i in range(5):
    print(
        f"Actual: {y_test.iloc[i]} | "
        f"Probability: {y_prob[i]:.2f} | "
        f"Risk Level: {risk_level(y_prob[i])}"
    )

# ==============================
# Phase 4: Explainability
# ==============================

try:
    import shap

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    print("\nTop Features Influencing Predictions:")
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Importance", ascending=False)

    print(feature_importance)

except Exception as e:
    print("\nSHAP not available, using model coefficients instead.\n")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print(coef_df)
