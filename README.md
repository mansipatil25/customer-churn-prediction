# 📊 Customer Churn Prediction

## Problem Statement
A telecom company losing 1 in 4 customers every month.
This project predicts which customers will churn and why.

## Dataset
- Source: Telco Customer Churn — Kaggle
- 7,032 customers | 21 features

## Tools Used
Python | pandas | scikit-learn | XGBoost | Power BI | matplotlib

## Project Steps
1. EDA — explored churn patterns
2. Data Cleaning — fixed data quality issues  
3. Feature Engineering — created 3 new features
4. Logistic Regression — 79.5% accuracy
5. XGBoost — caught 299/374 churners (79.9%)
6. Power BI Dashboard — interactive business visuals

## Key Findings
| Segment | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| New customers ≤12 months | 47.7% |
| Fiber optic users | 41.9% |
| Two-year contract | 2.8% |

## Model Results
| Model | Accuracy | Churners Caught |
|---|---|---|
| Logistic Regression | 79.5% | 199/374 |
| XGBoost | 72.4% | 299/374 |

## Business Recommendations
1. Target month-to-month customers with loyalty discounts
2. Add onboarding support in first 3 months
3. Investigate fiber optic pricing
4. Push automatic payment adoption

## 📸 Dashboard Preview
![Dashboard](telco%20data%20dashboard.png)
