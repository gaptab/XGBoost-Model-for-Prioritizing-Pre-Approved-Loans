import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Dummy Data
np.random.seed(42)
num_samples = 5000

data = pd.DataFrame({
    'customer_id': range(1, num_samples + 1),
    'age': np.random.randint(21, 65, num_samples),
    'income': np.random.randint(25000, 150000, num_samples),
    'credit_score': np.random.randint(300, 850, num_samples),
    'loan_amount': np.random.randint(1000, 50000, num_samples),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], num_samples),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], num_samples),
    'loan_status': np.random.choice(['Paid', 'Defaulted', 'Ongoing'], num_samples),
    'active_loans': np.random.randint(0, 5, num_samples),
    'default_risk': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),  # 10% risk level
    'pre_approved': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Target variable
})

# Step 2: Encode Categorical Features
label_enc = LabelEncoder()
data['employment_status'] = label_enc.fit_transform(data['employment_status'])
data['loan_status'] = label_enc.fit_transform(data['loan_status'])

# Step 3: Define Features & Target Variable
X = data.drop(columns=['customer_id', 'pre_approved'])  # Features
y = data['pre_approved']  # Target

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

# Step 7: Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Step 8: Visualization of Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Blues_r')
plt.title("Feature Importance in XGBoost Loan Approval Model")
plt.show()

# Step 9: Save Model & Data for Deployment
data.to_csv("pre_approved_loans_data.csv", index=False)
joblib.dump(model, "pre_approved_loans_xgb.pkl")

print("Dataset and Model Saved Successfully!")
