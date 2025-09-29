# src/train_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

print("🚀 Training script started...")

# 1. Load train/test data
X_train = pd.read_csv("data/X_train_pca.csv")
X_test = pd.read_csv("data/X_test_pca.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print(f"✅ Data loaded! Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Train Logistic Regression
print("\n🚀 Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f"✅ Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(classification_report(y_test, log_reg_pred))

# 3. Train Random Forest
print("\n🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

# 4. Train XGBoost
print("\n🚀 Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"✅ XGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred))

# 5. Pick best model
accuracies = {
    "LogReg": log_reg_acc,
    "RandomForest": rf_acc,
    "XGBoost": xgb_acc
}
best_model_name = max(accuracies, key=accuracies.get)
best_model = {"LogReg": log_reg, "RandomForest": rf, "XGBoost": xgb}[best_model_name]

# Ensure models folder exists
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print(f"\n🎉 Best model saved in models/best_model.pkl (Accuracy: {accuracies[best_model_name]:.4f})")

# 6. Save all results in reports/model_results.json
os.makedirs("reports", exist_ok=True)
results = {
    "LogReg": log_reg_acc,
    "RandomForest": rf_acc,
    "XGBoost": xgb_acc
}
with open("reports/model_results.json", "w") as f:
    json.dump(results, f)

print("✅ Model results saved in reports/model_results.json")