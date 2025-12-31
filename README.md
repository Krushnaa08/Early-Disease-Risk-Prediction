# Early Disease Risk Prediction

A machine learning project that predicts the risk of diabetes based on patient data. Built with Python, Scikit-learn, and Streamlit.

**INTRODUCTION**-----

Early detection of diabetes is an important challenge in preventive healthcare, particularly when decisions must be made based on limited and structured clinical data. This project was undertaken to explore the effectiveness of classical machine learning methods for early disease risk prediction and to better understand their behavior in a medical context.

Using a publicly available diabetes dataset, I implemented and compared multiple supervised learning models, including Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. The focus was placed not only on predictive accuracy but also on interpretability, robustness, and reproducibility of results—factors that are especially relevant in biomedical research.

The project follows a complete experimental pipeline, covering data preprocessing, feature scaling, model training, and evaluation using standard performance metrics. By comparing multiple models under the same experimental setup, this work provides insight into the strengths and limitations of different approaches for structured healthcare data.

Through this project, I aimed to build a solid foundation in applied machine learning research and to develop skills relevant to interdisciplinary research environments

## Overview
Early identification of disease risk plays a crucial role in preventive healthcare.  
This project implements and compares **classical machine learning models** to predict disease risk using structured medical data. The focus is on building **interpretable, efficient, and reproducible models** rather than overly complex architectures.

The project also serves as a small-scale research-oriented study comparing linear and non-linear classical ML methods.

---

## Models Implemented
The following machine learning models were trained and evaluated:

- **Logistic Regression**  
  A linear baseline model widely used in medical research for its interpretability.

- **K-Nearest Neighbors (KNN)**  
  A distance-based non-parametric model used to capture local patterns in the data.

- **Random Forest Classifier**  
  An ensemble-based method used to model non-linear relationships and improve robustness.

---

## Dataset
- Publicly available **diabetes dataset**
- 768 patient records
- 8 numerical medical features
- Binary target indicating disease risk

### Preprocessing Steps
- Handling missing or invalid values
- Feature standardization using `StandardScaler`
- Train-test split (80% training, 20% testing)

---

## Project Structure
Early-Disease-Risk-Prediction/
├── app.py # Streamlit web application
├── model_training.py # Model training and evaluation script
├── diabetes.csv # Dataset
├── best_model.pkl # Saved Logistic Regression model
├── scaler.pkl # Saved feature scaler
├── research project.md # Research paper (comparative study)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 2. Train Models
python model_training.py
-------------------------------------------------------------------------------------------------------------------------------------------------
## This will:

Train Logistic Regression, KNN, and Random Forest models
Print evaluation accuracy
Save the best model and scaler
-------------------------------------------------------------------------------------------------------------------------------------------------
## 3. Run the Web App
streamlit run app.py

-------------------------------------------------------------------------------------------------------------------------------------------------

## Experimental Results
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 81.2%    |
| Random Forest       | 78.6%    |
| KNN (k = 5)         | 72.7%    |

**Key Observation:**  
Despite its simplicity, Logistic Regression achieved the highest accuracy, indicating that classical linear models can perform competitively on structured medical datasets with limited size.

---

## How to Run the Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
-------------------------------------------------------------------------------------------------------------------------------------------------

## Research Component

This project is accompanied by a research-style comparative study:---

📄 research report.md
Title: Comparative Analysis of Classical Machine Learning Models for Non-Clinical Disease Risk Prediction

## The paper includes:

Abstract
Introduction
Dataset description
Methodology
Experimental setup
Results & discussion
Conclusion and future work
-------------------------------------------------------------------------------------------------------------------------------------------------
## Skills Demonstrated ---
Machine Learning model development

Model comparison and evaluation

Feature preprocessing and scaling

Research-oriented experimentation

Reproducible ML pipelines

GitHub project structuring
