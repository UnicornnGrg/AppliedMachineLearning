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

# --- Descriptive Labels for Categorical Variables ---
gender_labels = {1: 'Male', 2: 'Female'}
race_labels = {
    1: 'White',
    2: 'Black',
    3: 'AI/AN',
    4: 'Chinese',
    5: 'Japanese',
    6: 'Asian/NH/PI',
    7: 'Other',
    8: 'Two+ races'
}

# Apply labels if columns exist
if 'SEX' in df.columns:
    df['SEX_label'] = df['SEX'].map(gender_labels)
if 'RAC1P' in df.columns:
    df['RAC1P_label'] = df['RAC1P'].map(race_labels)

# --- Demographics Breakdown (Age, Gender, Race) ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Age Distribution
sns.histplot(df['AGEP'], bins=20, ax=axes[0], kde=True, color='skyblue')
axes[0].set_title('Distribution of Age (18-65)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Gender Distribution
if 'SEX_label' in df.columns:
    sns.countplot(x='SEX_label', data=df, ax=axes[1], palette='pastel')
    axes[1].set_title('Distribution of Gender')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Count')

# Race Distribution
if 'RAC1P_label' in df.columns:
    sns.countplot(x='RAC1P_label', data=df, ax=axes[2], palette='Set2', order=race_labels.values())
    axes[2].set_title('Distribution of Race')
    axes[2].set_xlabel('Race')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout()
plt.savefig("01 data_understanding/output/da_demographics_breakdown.png", dpi=200, bbox_inches='tight')
plt.close()

# --- Healthcare Analysis ---
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

def get_combination(row):
    selected = [hins_descriptions[col] for col in hins_cols if row[col] == 1]
    return ', '.join(selected) if selected else 'No coverage at all'

df['hins_combination'] = df[hins_cols].apply(get_combination, axis=1)
combination_counts = df['hins_combination'].value_counts()
combination_percentages = (combination_counts / len(df)) * 100
top_10_combinations = combination_percentages.nlargest(10)

# Top 10 HINS combinations as column chart
plt.figure(figsize=(14, 8))
top_10_combinations.plot(kind='bar', color=sns.color_palette('husl', len(top_10_combinations)))
plt.title('Top 10 Healthcare Coverage Combinations (% of Population 18-65)')
plt.xlabel('Healthcare Coverage Combination')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_top10_hins_combinations.png", dpi=200, bbox_inches='tight')
plt.close()

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
status_percentages = (df['insurance_status'].value_counts(normalize=True) * 100).sort_values(ascending=False)

# Insurance status as column chart
plt.figure(figsize=(10, 6))
status_percentages.plot(kind='bar', color=sns.color_palette('pastel', 4))
plt.title('Distribution of Public vs. Private Insurance Status (% of Population 18-65)')
plt.xlabel('Insurance Status')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_insurance_status.png", dpi=200, bbox_inches='tight')
plt.close()

bins = [18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 66]
labels = ['18-22', '23-27', '28-32', '33-37', '38-42', '43-47', '48-52', '53-57', '58-62', '63-65']
df['age_group'] = pd.cut(df['AGEP'], bins=bins, labels=labels, right=False)

medicaid_by_age = df.groupby('age_group')['HINS4'].apply(lambda x: (x == 1).mean() * 100)

plt.figure(figsize=(14, 7))
medicaid_by_age.plot(kind='bar', color='teal')
plt.title('Medicaid Coverage Rate by Age Group (% of Age Group)')
plt.xlabel('Age Group')
plt.ylabel('Medicaid Coverage Rate (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_medicaid_by_age.png", dpi=200, bbox_inches='tight')
plt.close()

# Print and save results
with open("01 data_understanding/output/healthcare_analysis.txt", "w") as f:
    f.write("Top 10 Healthcare Coverage Combinations (% of Population 18-65):\n")
    f.write(top_10_combinations.to_string())
    f.write("\n\nDistribution of Public vs. Private Insurance Status (% of Population 18-65):\n")
    f.write(status_percentages.to_string())
    f.write("\n\nMedicaid Coverage Rate by Age Group (% of Age Group):\n")
    f.write(medicaid_by_age.to_string())

print("Analysis complete. Outputs saved to '01 data_understanding/output/'.")
