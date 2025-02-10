# XGBoost-Model-for-Prioritizing-Pre-Approved-Loans

![alt text](https://github.com/gaptab/XGBoost-Model-for-Prioritizing-Pre-Approved-Loans/blob/main/472.png)

**Generating Data**

Creates 5,000 synthetic customer records with details like age, income, credit score, loan amount, employment status, loan history, and default risk.

The target variable is pre_approved (1 = loan pre-approved, 0 = not pre-approved).

Maintains 10% risk level (default_risk = 1 for 10% of cases).

**Data Preprocessing**

Encodes categorical variables (e.g., employment_status, loan_status) into numerical values.

Drops customer_id since it’s not useful for prediction.

**Splitting Data into Train & Test Sets**

Splits data into 80% training and 20% testing to evaluate model performance.

**Training the XGBoost Model**

Uses XGBoost Classifier with:

100 estimators (trees)

Max depth = 5 (to balance complexity and performance)

Learning rate = 0.1 (controls model updates)

**Model Evaluation**

Predicts on the test set and calculates:
Accuracy
Precision, Recall, and F1-score (for both approved & non-approved cases).

Expected accuracy: ~85-90%.

**Feature Importance Analysis**

Extracts which features influence loan approval the most.

Typically, credit score, income, loan status, and employment status are the most critical factors.

Visualizes feature importance using a bar chart.
