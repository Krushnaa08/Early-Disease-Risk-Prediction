## **Comparative Analysis of Classical Machine Learning Models for Non-Clinical Disease Risk Prediction**

## **Abstract**

Early detection of chronic diseases plays a critical role in preventive healthcare. Machine learning techniques offer the potential to assist medical professionals by identifying disease risks using patient health indicators. In this study, we compare classical machine learning models—Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest—for early disease risk prediction using a structured medical dataset. The models are evaluated using accuracy as the primary performance metric. Logistic Regression was ultimately selected for deployment due to its balanced performance and interpretability, while KNN and Random Forest were used as comparative baselines. The results demonstrate that simple, interpretable models can perform competitively with more complex methods in healthcare-related prediction tasks.

## **1\. Introduction**

With the increasing availability of medical data, machine learning has become an important tool in healthcare decision support systems. Predicting disease risk at an early stage can help reduce long-term treatment costs and improve patient outcomes. While modern deep learning approaches have gained attention, classical machine learning models remain highly relevant, especially in medical applications where interpretability and reliability are essential.

This work focuses on evaluating commonly used classical machine learning algorithms for disease risk prediction. Instead of proposing a new algorithm, the goal of this study is to systematically compare multiple well-established models under the same experimental setup. Such comparisons are useful for understanding the trade-offs between performance, simplicity, and interpretability. Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest classifiers are selected for this study due to their frequent use in medical data analysis.

## **2\. Dataset Description**

The dataset used in this study is a publicly available medical dataset containing patient health records. Each instance represents a patient, described by numerical health indicators such as glucose level, blood pressure, body mass index (BMI), and age. The target variable is binary, indicating whether the patient is at risk of developing a particular disease.

Before training, the dataset was inspected for missing values and inconsistencies. Basic preprocessing steps were applied to ensure data quality. Since the features are numerical and vary in scale, feature standardization was performed for models sensitive to distance calculations, such as KNN.

---

## **3\. Methodology**

This study evaluates three classical machine learning models:

* **Logistic Regression**: A linear model commonly used in medical research due to its interpretability and probabilistic output.

* **K-Nearest Neighbors (KNN)**: A distance-based model that classifies a sample based on the majority class among its nearest neighbors.

* **Random Forest**: An ensemble learning method that combines multiple decision trees to improve prediction robustness.

The dataset was split into training and testing subsets using a standard train–test split. Feature scaling was applied using standardization to ensure fair comparison, particularly for KNN. Each model was trained using the same training data and evaluated on the same test set.

---

## **4\. Experimental Setup**

* **Programming Language**: Python

* **Libraries Used**: scikit-learn, pandas, numpy

* **Train–Test Split**: 80% training, 20% testing

* **Evaluation Metric**: Accuracy

For KNN, the number of neighbors was set to *k \= 5* based on common practice. Random Forest was trained using default hyperparameters to maintain fairness and avoid overfitting due to excessive tuning. Logistic Regression was trained using standard regularization settings.

---

## **5\. Results and Discussion**

The experimental results indicate that all three models achieve comparable performance, with Logistic Regression and Random Forest slightly outperforming KNN. KNN showed lower accuracy, which may be attributed to its sensitivity to noise and reliance on distance metrics in higher-dimensional feature space.

Despite Random Forest achieving competitive accuracy, Logistic Regression was chosen as the final model for deployment. This decision was based not only on accuracy but also on model interpretability, which is especially important in healthcare applications where understanding model decisions is critical.

The results suggest that simpler models can be effective for disease risk prediction and that model selection should consider interpretability alongside performance metrics.

---![model_training](https://github.com/user-attachments/assets/2665a00c-3582-425c-84e2-9474c9fd4975)


## **6\. Conclusion and Future Work**

This study presented a comparative analysis of classical machine learning models for early disease risk prediction. Logistic Regression, K-Nearest Neighbors, and Random Forest classifiers were evaluated using a consistent experimental framework. Logistic Regression demonstrated stable performance while offering the advantage of interpretability, making it suitable for deployment in a healthcare-oriented application.

Future work may include evaluating additional models such as Support Vector Machines or gradient boosting methods. Incorporating cross-validation, hyperparameter tuning, and additional evaluation metrics such as precision and recall could further strengthen the analysis. Moreover, testing the approach on larger and more diverse medical datasets would improve generalizability.

