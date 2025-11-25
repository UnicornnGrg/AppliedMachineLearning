import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the output directory exists
os.makedirs("01 data_understanding/output", exist_ok=True)

# Load the raw data
print("Loading data...")
df = pd.read_csv('00 data/raw/psam_p38.csv')

# Filter to ages 18-65
df = df[(df['AGEP'] >= 18) & (df['AGEP'] <= 65)]

# Save dataset overview to a text file
with open("01 data_understanding/output/dataset_overview.txt", "w") as f:
    f.write(f"{'='*80}\n")
    f.write("DATASET OVERVIEW\n")
    f.write(f"{'='*80}\n")
    f.write(f"Number of rows: {df.shape[0]:,}\n")
    f.write(f"Number of columns: {df.shape[1]}\n")
    f.write(f"\n{'='*80}\n")
    f.write("COLUMN NAMES\n")
    f.write(f"{'='*80}\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:3d}. {col}\n")
    f.write(f"\n{'='*80}\n")
    f.write("DATA TYPES\n")
    f.write(f"{'='*80}\n")
    f.write(df.dtypes.to_string())
    f.write(f"\n{'='*80}\n")
    f.write("MISSING VALUES\n")
    f.write(f"{'='*80}\n")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    if len(missing_df) > 0:
        f.write(missing_df.to_string(index=False))
    else:
        f.write("No missing values found!")
    f.write(f"\n{'='*80}\n")
    f.write("FIRST 5 ROWS\n")
    f.write(f"{'='*80}\n")
    f.write(df.head().to_string())
    f.write(f"\n{'='*80}\n")
    f.write("BASIC STATISTICS\n")
    f.write(f"{'='*80}\n")
    f.write(df.describe().to_string())
    f.write(f"\n{'='*80}\n")
    f.write("UNIQUE VALUES PER COLUMN (for categorical-looking columns)\n")
    f.write(f"{'='*80}\n")
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique < 20:
            f.write(f"\n{col}: {n_unique} unique values\n")
            f.write(df[col].value_counts().head(10).to_string())

# Healthcare columns and their descriptive names
hins_cols = ['HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7']
hins_descriptions = {
    'HINS1': 'Employer-based',
    'HINS2': 'Direct-purchase',
    'HINS3': 'Medicare',
    'HINS4': 'Medicaid',
    'HINS5': 'Tricare/VA',
    'HINS6': 'Other public',
    'HINS7': 'No coverage'
}

# --- Combinations of HINS1-HINS7 ---
def get_combination(row):
    selected = [hins_descriptions[col] for col in hins_cols if row[col] == 1]
    return ', '.join(selected) if selected else 'No coverage at all'

df['hins_combination'] = df[hins_cols].apply(get_combination, axis=1)
combination_counts = df['hins_combination'].value_counts()
combination_percentages = (combination_counts / len(df)) * 100
top_10_combinations = combination_percentages.nlargest(10)

# Plot top 10 HINS combinations
plt.figure(figsize=(14, 8))
top_10_combinations.plot(kind='barh', color=sns.color_palette('husl', len(top_10_combinations)))
plt.title('Top 10 Healthcare Coverage Combinations (% of Total)')
plt.xlabel('Percentage (%)')
plt.ylabel('Healthcare Coverage Combination')
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_top10_hins_combinations.png", dpi=200, bbox_inches='tight')
plt.close()

# --- Public/Private/Both/None Insurance Status ---
def get_insurance_status(row):
    public = any(row[['HINS3', 'HINS4', 'HINS5', 'HINS6']] == 1)
    private = any(row[['HINS1', 'HINS2']] == 1)
    if public and private:
        return 'Both public and private'
    elif public:
        return 'Public only'
    elif private:
        return 'Private only'
    else:
        return 'Not insured'

df['insurance_status'] = df.apply(get_insurance_status, axis=1)
status_percentages = (df['insurance_status'].value_counts(normalize=True) * 100).sort_values()

# Plot insurance status distribution
plt.figure(figsize=(10, 6))
status_percentages.plot(kind='barh', color=sns.color_palette('pastel', 4))
plt.title('Public vs. Private Insurance Status (% of Total)')
plt.xlabel('Percentage (%)')
plt.ylabel('Insurance Status')
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_insurance_status.png", dpi=200, bbox_inches='tight')
plt.close()

# Print and save results
with open("01 data_understanding/output/healthcare_analysis.txt", "w") as f:
    f.write("Top 10 Healthcare Coverage Combinations (% of Total):\n")
    f.write(top_10_combinations.to_string())
    f.write("\n\nPublic vs. Private Insurance Status (% of Total):\n")
    f.write(status_percentages.to_string())

print("Analysis complete. Outputs saved to '01 data_understanding/output/'.")
