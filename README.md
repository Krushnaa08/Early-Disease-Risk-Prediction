# AI-Based Early Disease Risk Prediction System

Predict diabetes risk using machine learning with an interactive **Streamlit web app**.

## Features
- Early disease risk prediction (non-clinical)
- Logistic Regression & Random Forest models
- Data preprocessing and scaling
- Risk-level output: Low / Medium / High
- Interactive web interface using Streamlit

## Dataset
Pima Indians Diabetes Database (768 samples, 8 features + target)

**Columns:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Installation
```bash
git clone <your-github-repo-link>
cd "first project"
python -m pip install -r requirements.txt
python -m streamlit run app.py
