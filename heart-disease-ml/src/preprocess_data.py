import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print(" Script started...")

# Step 1: Load Data
print("Step 1 Loading data...")
data = pd.read_csv("data/heart.csv")
print(f" Data loaded with shape: {data.shape}")

# Step 2: Drop unnecessary columns
print("Step 2 Dropping unnecessary columns...")
X = data.drop(["id", "num"], axis=1)
y = data["num"]
print(f" X shape: {X.shape}, y shape: {y.shape}")

# Step 3: Split
print("Step 3 Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Step 4: Preprocessing pipeline
print("Step 4 Building preprocessor...")
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit
print("Step 5 Fitting preprocessor...")
pipeline.fit(X_train)
print(" Preprocessor fitted!")

# Transform
print("Step 6 Transforming data...")
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)
print(f" X_train transformed shape: {X_train_transformed.shape}")
print(f" X_test transformed shape: {X_test_transformed.shape}")

# Save processed data
print("Step 7 Saving processed data...")
pd.DataFrame(X_train_transformed).to_csv("data/X_train_processed.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("data/X_test_processed.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print(" Processed data saved in /data folder.")

# Save pipeline
print("Step 8  Saving pipeline...")
joblib.dump(pipeline, "models/preprocessor.pkl")
print(" Pipeline saved in /models folder.")

print(" Preprocessing completed successfully!")