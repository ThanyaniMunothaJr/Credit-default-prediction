# Credit-default-prediction
Predict whether a borrower will default on a loan using historical credit data. (Binary classification)

## Overview

This project builds a **credit risk classification model** to predict whether a loan will **default(1)** or be **fully repaid(0)** using historical LendingClub data (2007–2015).

The project is intentionally structured in **two parts**:

* **Part 1**: Interpretable baseline model using Logistic Regression
* **Part 2**: Tree-based models and performance improvements

The focus of Part 1 is **correct modeling practice**, metric selection, and clear financial interpretation rather than maximum performance.

---

## Problem Statement

Given borrower and loan characteristics at origination, can we predict whether a loan will default?

This is a standard **binary classification** problem in credit risk, where:

* `0` = Non-default (Fully Paid)
* `1` = Default (Charged Off)

Because defaults are **rare events**, the dataset is **class-imbalanced**, which strongly influences model evaluation and metric choice.

---

## Dataset

* **Source**: LendingClub public loan data
* **Time Period**: 2007–2015
* **Target Variable**: Loan default status
* **Classes**:
  
  * Fully Paid (majority class)
  * Charged Off (minority class)

To improve the computing speed we used the first 1,000,000 entries of the dataset as this dataset is extremely large. This subset of entries is still large enough to give us a valid model.

Only loans with a clear final outcome were retained to avoid label noise.

---

## Feature Engineering

Key preprocessing steps included:

* Converting loan term from string to numeric (e.g. "36 months" → 36)
* Creating **credit history length (years)** from earliest credit line date
* Selecting financially meaningful numerical and categorical features
* Handling missing values using **imputation within the modeling pipeline**

This approach avoids data leakage and mirrors real-world production pipelines.

---

## Train / Test Split

* Stratified train-test split
* Preserves default / non-default proportions
* Prevents biased evaluation due to class imbalance

---

## Model (Part 1 – Baseline)

### Logistic Regression

Logistic Regression was chosen as the baseline model because:

* It is widely used in credit risk
* Coefficients are interpretable
* It provides a strong linear benchmark

The model was trained using a scikit-learn **Pipeline** combining:

* Imputation
* Scaling
* One-hot encoding
* Logistic Regression classifier

---

## Evaluation Metrics

### Why Accuracy Is Not Enough

Due to class imbalance, accuracy can be misleading. A naive model predicting "no default" for all loans would achieve high accuracy but zero usefulness.

### Primary Metric: ROC-AUC

* Measures ranking quality across all thresholds
* Standard metric in credit risk modeling
* Threshold-independent

### Secondary Metrics

* **Recall (Default Class)**: Ability to identify high-risk borrowers
* **Precision (Default Class)**: Cost of false positives

---

## Results (Part 1)

* **Model**: Logistic Regression
* **ROC-AUC (Test Set)**: **0.6985974162001144**
* **Accuracy**: **0.7997688470222434**

### Interpretation

An ROC-AUC of 0.6985974162001144 indicates that the model meaningfully distinguishes between risky and safe borrowers and performs substantially better than random classification.

Recall for the default class shows that the model captures a non-trivial portion of high-risk loans, which is more important than overall accuracy in credit-risk applications.

This model serves as a strong, interpretable baseline.

---

## Limitations

* Linear decision boundary
* Limited interaction effects
* No hyperparameter tuning

These limitations will be addressed in **Part 2**.

---

## Part 2
Planned improvements include:

* Random Forest and Gradient Boosting models
* Feature importance analysis
* Improved handling of class imbalance
* Performance comparison across models

---

In the final notebook I was able to utilize the entire dataset and not trim to the frist 1,000,000 entries due to an increase in memory capacity of my computer(Yes, I upgraded). You will also spot some small changes in the initial code to make it all cleaner.

---

## Results part 2

* **Model**: Tree based model - Random Forest
* **ROC-AUC (Test Set)**: **0.7052853956612875**
* **Accuracy**: **0.8006689117143931**

---

## Feature Importance

Feature importance from the Random Forest model highlights the primary drivers of default risk.

Interest rate and credit grade are the most influential variables consistent with risk-based loan pricing. Term length and dti ratio also plaay significant roles, reflecting borrower leverage and credit maturity.


---

## Use case

In this notebook we will see how this model works when faced with an individual looking to take out a loan or has already taken out a loan and needs to be assessed for further risk management/planning to take place.

Let's create a useable tool for people to input their metrics and get a result of whether our financial institution will see them as likely to default or not...

----

This tool will enable them to:
- See how varioius loan providers view them from behind the desk
- Play with the metrics and see what numbers can take them from being a "likely to default" client to one that is not
- Be able to set targets for themselves to reach in order to be more confident when applying for loans

-------

Ethical use for a tool such as this will be of high importance in determining whether it can be launched for the general public. This model is not perfect. Work still needs to be done to improve its performance to a level where the output will be usable. I am open to collaborate with all bright minds to take this model to the next level and develop methodologies to ensure that this tool will be used ethically in the future. Happy modelling!

---

## END

---
## Skills Used

* Credit risk modeling
* Handling class imbalance
* Proper metric selection (ROC-AUC)
* Feature engineering
* Scikit-learn pipelines
* Reproducible ML workflows

