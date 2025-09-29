import pandas as pd

df = pd.read_csv('data/heart.csv', encoding='utf-8')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head().to_string(index=False))
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())

# check target value distribution
if 'target' in df.columns:
    print("\nTarget distribution:\n", df['target'].value_counts())
elif 'num' in df.columns:
    print("\nNum distribution:\n", df['num'].value_counts())