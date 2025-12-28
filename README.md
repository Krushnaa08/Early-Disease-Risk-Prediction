# Early Disease Risk Prediction

A machine learning project that predicts the risk of diabetes based on patient data. Built with Python, Scikit-learn, and Streamlit.

## 🔹 Features

- Logistic Regression model for diabetes prediction
- Handles missing values and scales data automatically
- Shows probability and risk level (Low/High Risk)
- Feature importance visualization
- Interactive web app using Streamlit

## 🔹 Project Structure

first-project/
├── app.py # Streamlit web app
├── model_training.py # Model training script
├── best_model.pkl # Saved Logistic Regression model
├── scaler.pkl # Saved StandardScaler
├── diabetes.csv # Dataset
├── requirements.txt # Python dependencies
├── screenshots/ # Sample screenshots of app
└── README.md # Project description


## 🔹 How to Run Locally

1. Clone this repository:
```bash
git clone <YOUR_REPO_URL>
cd first-project
Install dependencies:

bash
Copy code
python -m pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
python -m streamlit run app.py
🔹 Sample Screenshots



🔹 Libraries Used
pandas

numpy

scikit-learn

joblib

streamlit

matplotlib / seaborn (if used)

🔹 Author
Krushna Salunkhe