import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print(" Training script started...")

# =========================
# Load PCA-transformed data
# =========================
X_train = pd.read_csv("data/X_train_pca.csv")
X_test = pd.read_csv("data/X_test_pca.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()



print(f" Data loaded! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =========================
# Define models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42)
}

best_model = None
best_acc = 0

# =========================
# Train and evaluate
# =========================
for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f" {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_acc:
        best_acc = acc
        best_model = model

# =========================
# Save best model
# =========================
if best_model:
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\n Best model saved in models/best_model.pkl (Accuracy: {best_acc:.4f})")
else:
    print(" No model was trained successfully.")