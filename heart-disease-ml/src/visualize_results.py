import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import json

print(" Visualization script started...")

# 1. Load test data
X_test = pd.read_csv("data/X_test_pca.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# 2. Load best model
model = joblib.load("models/best_model.pkl")
print(" Best model loaded!")

# 3. Predictions
y_pred = model.predict(X_test)

# -------------------------------
#  Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/confusion_matrix.png")
plt.show()

# -------------------------------
#  ROC Curve (multi-class)
# -------------------------------
n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
y_score = model.predict_proba(X_test)

plt.figure(figsize=(7,5))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multi-class)")
plt.legend(loc="lower right")
plt.savefig("reports/roc_curve.png")
plt.show()

# -------------------------------
#  Accuracy Bar Chart
# -------------------------------
# Assume that in train_models.py you stored the results in a JSON file
#example {"LogReg": 0.55, "RandomForest": 0.57, "XGBoost": 0.56}

with open("reports/model_results.json", "r") as f:
    results = json.load(f)

models = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.ylim(0,1)
plt.title("Model Accuracies")
plt.ylabel("Accuracy")
plt.savefig("reports/model_accuracies.png")
plt.show()

print(" All visualizations saved in reports/ folder")