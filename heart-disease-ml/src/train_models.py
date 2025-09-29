import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import json

print(" Training script started...")

# 1. Load raw train/test data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print(f" Raw Data loaded! Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Preprocessing: Scaling + PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(" Scaling and PCA applied!")

# 3. Train Logistic Regression
print("\n Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_pca, y_train)
log_reg_pred = log_reg.predict(X_test_pca)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f" Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(classification_report(y_test, log_reg_pred))

# 4. Train Random Forest
print("\n Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
rf_pred = rf.predict(X_test_pca)
rf_acc = accuracy_score(y_test, rf_pred)
print(f" Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

# 5. Train XGBoost
print("\n Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb.fit(X_train_pca, y_train)
xgb_pred = xgb.predict(X_test_pca)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f" XGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred))

# 6. Pick best model
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
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")

print(f"\n Best model saved in models/best_model.pkl (Accuracy: {accuracies[best_model_name]:.4f})")
print(" Scaler and PCA saved in models/")

# 7. Save all results in reports/model_results.json
os.makedirs("reports", exist_ok=True)
with open("reports/model_results.json", "w") as f:
    json.dump(accuracies, f)

print(" Model results saved in reports/model_results.json")