# Credit Risk Assessment: Loan Default Prediction

## Overview

Developed an end-to-end machine learning pipeline for **credit risk assessment and loan default prediction** using the German Credit Dataset. The project focuses on building interpretable classification models and optimizing lending decisions based on the asymmetric costs of credit risk.

### Highlights

- Analyzed **1,000 credit applicants** across **20 applicant attributes**
- Performed data preprocessing, categorical encoding, and risk-oriented feature engineering
- Developed and compared **6 classification models**
- Evaluated models using **5-fold stratified cross-validation** and an **80/20 train-test split**
- Implemented **cost-sensitive classification** to account for the higher cost of approving potential defaulters
- Used **SHAP-based interpretability** to identify key drivers of credit risk
- Achieved a **Test ROC-AUC of 80.43%** with Logistic Regression
- Reduced classification cost by **38.4%** through threshold optimization

---

## Dataset

The project uses the **German Credit Dataset** from the UCI Machine Learning Repository.

| Attribute | Value |
|---|---|
| Applicants | 1,000 |
| Features | 20 |
| Numerical Features | 7 |
| Categorical Features | 13 |
| Target | Good Credit / Bad Credit |
| Class Distribution | 70% Good / 30% Bad |

Key variables include checking account status, credit history, loan duration, credit amount, savings, employment, housing, age, and existing credits.

---

## Methodology

### 1. Data Preprocessing

- Decoded categorical variables for improved interpretability
- Converted the target into a binary classification variable
- Processed numerical and categorical variables using a unified preprocessing pipeline
- Prepared the dataset for model training and evaluation

### 2. Feature Engineering

Engineered additional risk-oriented features including:

- Credit-to-income ratio
- Estimated monthly payment
- Employment stability indicators
- Housing stability indicators

### 3. Model Development

Implemented and evaluated six classification algorithms:

- Logistic Regression
- Random Forest
- LightGBM
- Gradient Boosting
- XGBoost
- Decision Tree

Models were evaluated using **ROC-AUC, F1-score, precision, recall, and confusion matrices**.

---

## Model Performance

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|---|---:|---:|---:|
| **Logistic Regression** | **0.7682** | **0.8043** | **0.6055** |
| Random Forest | 0.7831 | 0.8037 | 0.5263 |
| LightGBM | 0.7900 | 0.7793 | 0.5766 |
| Gradient Boosting | 0.7767 | 0.7835 | 0.5636 |
| XGBoost | 0.7664 | 0.7512 | 0.5546 |
| Decision Tree | 0.6753 | 0.7090 | 0.4348 |

**Logistic Regression** was selected as the final model based on its predictive performance and interpretability, achieving a **Test ROC-AUC of 80.43%**.

---

## Cost-Sensitive Optimization

Credit decisions have asymmetric consequences: approving a defaulter can result in substantially higher losses than rejecting a potentially good applicant.

A custom cost matrix was incorporated into the decision process to optimize the classification threshold.

| Threshold | False Positives | False Negatives | Recall | Total Cost |
|---|---:|---:|---:|---:|
| 0.50 | 16 | 27 | 55.0% | 151 |
| **0.30** | **38** | **11** | **81.7%** | **93** |

Optimizing the classification threshold from **0.50 to 0.30** increased default detection recall from **55.0% to 81.7%** and reduced the overall classification cost by **38.4%**.

---

## Model Interpretability

Used **SHAP (SHapley Additive exPlanations)** to interpret model predictions and identify the features contributing most significantly to credit risk.

Key risk drivers identified include:

- Checking account status
- Credit history
- Loan duration
- Credit amount
- Savings status
- Employment stability

SHAP analysis was used to provide both **global feature importance** and **individual applicant-level explanations**.

---

## Key Findings

- **Checking account status** emerged as one of the strongest indicators of default risk.
- **Longer loan durations** were associated with increased credit risk.
- **Historical credit behavior** provided significant predictive information about future defaults.
- Cost-sensitive threshold optimization substantially improved the model's ability to detect potential defaults.
- SHAP analysis improved model transparency and supported interpretation of individual credit decisions.

---

## Risk Segmentation

The model can be used to segment applicants based on predicted default probability and support differentiated lending decisions.

| Risk Level | Default Probability | Suggested Action |
|---|---:|---|
| Very Low | <20% | Auto-approve |
| Low | 20–40% | Standard approval |
| Medium | 40–60% | Enhanced review |
| High | 60–80% | Decline / Collateral |
| Very High | >80% | Decline |

---

## Project Structure

```text
credit-risk-assessment/
│
├── data/
│   ├── raw/
│   │   └── german_credit.data
│   └── processed/
│       └── german_credit_processed.csv
│
├── docs/
│   ├── analysis_report.md
│   └── figures/
│       ├── target_distribution.png
│       ├── numerical_distributions.png
│       ├── categorical_default_rates.png
│       ├── correlation_matrix.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       ├── threshold_optimization.png
│       ├── shap_importance.png
│       ├── shap_beeswarm.png
│       └── risk_distribution.png
│
├── models/
│   ├── credit_risk_model.joblib
│   └── preprocessor.joblib
│
├── notebooks/
│   └── credit_risk_assessment.ipynb
│
├── src/
│   └── predict_risk.py
│
├── requirements.txt
└── README.md

