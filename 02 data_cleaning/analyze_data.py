import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
df = pd.read_csv('data/raw/psam_p38.csv')

print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print(f"{'='*80}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print(f"\n{'='*80}")
print("COLUMN NAMES")
print(f"{'='*80}")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

print(f"\n{'='*80}")
print("DATA TYPES")
print(f"{'='*80}")
print(df.dtypes)

print(f"\n{'='*80}")
print("MISSING VALUES")
print(f"{'='*80}")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values found!")

print(f"\n{'='*80}")
print("FIRST 5 ROWS")
print(f"{'='*80}")
print(df.head())

print(f"\n{'='*80}")
print("BASIC STATISTICS")
print(f"{'='*80}")
print(df.describe())

print(f"\n{'='*80}")
print("UNIQUE VALUES PER COLUMN (for categorical-looking columns)")
print(f"{'='*80}")
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique < 20:  # Only show columns with less than 20 unique values
        print(f"\n{col}: {n_unique} unique values")
        print(df[col].value_counts().head(10))
