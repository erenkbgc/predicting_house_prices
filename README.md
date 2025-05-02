# House Price Prediction – Advanced Regression Techniques

This repository contains a complete machine learning pipeline for predicting house sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview). The goal is to build a well-generalized regression model that incorporates structured data preprocessing, regularization, and ensemble learning.

The project includes all major stages of applied machine learning: data cleaning, feature engineering, model comparison, hyperparameter tuning, and performance evaluation.

---

## Project Overview

- Data cleaning and missing value imputation  
- Outlier treatment using the IQR method  
- Feature encoding and scaling  
- Log transformation of the target variable to reduce skew  
- Model building and evaluation using:  
  - Linear Regression  
  - Ridge Regression  
  - XGBoost Regressor  
- Hyperparameter tuning with GridSearchCV  
- Model evaluation using Root Mean Squared Logarithmic Error (RMSLE)  
- Feature importance analysis for interpretability

---

## Repository Contents

- `house-pricing.ipynb` — Complete Jupyter notebook with code, explanations, and results  
- `submission.csv` — Final predictions formatted for Kaggle submission  
- `house-pricing-final.html` — Exported version of the notebook for viewing without running code  
- `README.md` — Project documentation

---

## Evaluation Metric

The project uses **Root Mean Squared Logarithmic Error (RMSLE)** as the evaluation metric. RMSLE penalizes underestimations more heavily and is particularly suitable for targets that vary across multiple orders of magnitude.

$$
\mathrm{RMSLE}(y, \hat{y}) = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
$$

---

## Results

After comparing multiple models and performing grid search for hyperparameter tuning, the best-performing model was **XGBoost**, which achieved a **cross-validated RMSLE of 0.122**.

---

## Getting Started

To run the notebook locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-pricing.git
   cd house-pricing
