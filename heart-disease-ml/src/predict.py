import pandas as pd
import joblib
import numpy as np

# 1. Load best model
model = joblib.load("models/best_model.pkl")
print("✅ Best model loaded!")

# 2. Example input as DataFrame with same feature names
# افتح ملف train عشان تجيب أسماء الأعمدة (X_train_pca.csv)
feature_names = pd.read_csv("data/X_train_pca.csv").columns

example_patient = pd.DataFrame(
    [[0.5, -1.2, 0.3, 0.8, -0.6, 
      1.1, -0.4, 0.7, -0.2, 0.9,
      -1.1, 0.4, -0.7, 0.2, 1.3]],
    columns=feature_names
)

# 3. Predict
prediction = model.predict(example_patient)[0]
print(f"🎯 Predicted Heart Disease Class: {prediction}")