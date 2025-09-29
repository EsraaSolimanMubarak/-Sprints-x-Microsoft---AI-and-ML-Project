import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("🧪 Evaluation script started...")

# 1. Load test data
X_test = pd.read_csv("data/X_test_pca.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()
print(f" Test data loaded! Shape: {X_test.shape}")

# 2. Load saved best model
model = joblib.load("models/best_model.pkl")
print(" Best model loaded from models/best_model.pkl")

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluation metrics
acc = accuracy_score(y_test, y_pred)
print(f"\n Test Accuracy: {acc:.4f}\n")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(len(set(y_test))),
            yticklabels=range(len(set(y_test))))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()