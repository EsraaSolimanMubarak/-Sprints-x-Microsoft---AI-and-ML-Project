import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("🔥 PCA script started...")

   # Upload processed data
    X_train = pd.read_csv("data/X_train_processed.csv")
    X_test = pd.read_csv("data/X_test_processed.csv")

    # Process any remaining NaN values ​​(by substituting the mean)
    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_test.mean(numeric_only=True))

    print("✅ Data loaded & cleaned!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # PCA
    pca = PCA(n_components=15)
    # pca.fit(X_train)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("✅ PCA applied!")
    print("Train shape after PCA:", X_train_pca.shape)
    print("Test shape after PCA:", X_test_pca.shape)

    #percentage of explained variance
    explained_variance = pca.explained_variance_ratio_

    #cumulative variance drawing
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_variance)+1),
             explained_variance.cumsum(), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Explained Variance")
    plt.grid(True)
    plt.show()
    pd.DataFrame(X_train_pca).to_csv("data/X_train_pca.csv", index=False)
    pd.DataFrame(X_test_pca).to_csv("data/X_test_pca.csv", index=False)


    print("📊 Explained variance ratio per component:")
    print(explained_variance)
    print("💾 PCA-transformed data saved in /data/")