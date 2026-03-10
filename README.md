# Customer Churn Prediction – End-to-End Machine Learning Project

## Overview
This project builds a machine learning system to predict whether a telecom customer will churn. Customer churn occurs when a customer stops using a company's services. Predicting churn helps companies identify at-risk customers and take proactive retention actions.

The project follows a full machine learning workflow including data analysis, preprocessing pipelines, model training, hyperparameter tuning, evaluation, and prediction.

---

## Problem Statement
Predict whether a customer will churn based on demographic information, service usage, and billing details.

Target Variable:

Churn  
1 → Customer will churn  
0 → Customer will stay

---

## Dataset
The dataset contains telecom customer information including:

Numerical Features
- tenure
- MonthlyCharges
- TotalCharges

Categorical Features
- Contract
- InternetService
- PaymentMethod
- OnlineSecurity
- TechSupport

Each row represents one customer.

---

## Machine Learning Workflow

### 1. Data Understanding
Initial analysis included:
- dataset inspection
- missing value detection
- class imbalance analysis
- feature distribution exploration

### 2. Data Preprocessing
A preprocessing pipeline was built using scikit-learn pipelines.

Steps included:
- Missing value imputation
- Standard scaling for numerical features
- One-hot encoding for categorical features

Implemented using:
- ColumnTransformer
- Pipeline
- SimpleImputer
- StandardScaler
- OneHotEncoder

### 3. Train/Test Split
The dataset was split using stratified sampling to maintain class balance.

train_test_split(test_size=0.2, stratify=y)

### 4. Baseline Model
A baseline model was trained using Logistic Regression.

### 5. Model Selection
The following models were evaluated:
- Logistic Regression
- Random Forest

Random Forest performed better for this dataset.

### 6. Hyperparameter Tuning
Model performance was improved using GridSearchCV.

Parameters tuned:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

Best parameters found:

max_depth = 15  
min_samples_split = 5  
min_samples_leaf = 1  
n_estimators = 200

---

## Model Evaluation

Evaluation metrics used:
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC-AUC

Results:

F1 Score ≈ 0.56  
Recall (Churn) ≈ 0.91  
ROC-AUC ≈ 0.83  

High recall ensures most churners are detected.

---

## Threshold Optimization
Since churn datasets are imbalanced, predictions were optimized using precision-recall analysis.

Best probability threshold:

0.529

Decision rule:

if churn_probability > 0.529 → predict churn

---

## Feature Importance

Top features influencing churn:

1. Contract type (Month-to-month)
2. Online security availability
3. Customer tenure
4. Total charges
5. Tech support availability
6. Payment method (Electronic check)
7. Monthly charges
8. Internet service type

These insights can help companies design customer retention strategies.

---

## Project Structure

churn_prediction_ml  
│  
├── data  
│   ├── raw  
│   └── processed  
│  
├── models  
│   └── churn_model.pkl  
│  
├── notebooks  
│   └── churn_analysis.ipynb  
│  
├── src  
│   ├── data  
│   │   ├── load_data.py  
│   │   └── preprocess.py  
│   │  
│   ├── models  
│   │   ├── train_model.py  
│   │   ├── evaluate_model.py  
│   │   └── predict.py  
│  
├── main.py
├── .gitignore    
├── requirements.txt  
└── README.md  

---

## Running the Project

### 1. Clone the Repository

git clone <repo-url>  
cd churn_prediction_ml  

### 2. Create Virtual Environment

python -m venv venv  

Activate environment:

venv\Scripts\activate  

### 3. Install Dependencies

pip install -r requirements.txt  

### 4. Train the Model

python main.py  

This will:
- train the model
- evaluate performance
- save the trained pipeline

Saved model:

models/churn_model.pkl  

### 5. Predict Churn for a New Customer

python src/models/predict.py  

Example output:

Churn Probability: 0.71  
Prediction: Customer likely to churn  

---

## Future Improvements

Possible extensions:

- gradient boosting models (XGBoost / LightGBM)
- additional feature engineering
- model monitoring
- API deployment for real-time predictions
