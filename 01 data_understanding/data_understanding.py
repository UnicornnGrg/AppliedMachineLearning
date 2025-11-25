import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the output directory exists
os.makedirs("01 data_understanding/output", exist_ok=True)

# Load the raw data
data = pd.read_csv("00 data/raw/psam_p38.csv")

# Filter to ages 18-65
data = data[(data['AGEP'] >= 18) & (data['AGEP'] <= 65)]

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

data['hins_combination'] = data[hins_cols].apply(get_combination, axis=1)
combination_counts = data['hins_combination'].value_counts()
combination_percentages = (combination_counts / len(data)) * 100
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
    public = any(row[['HINS3', 'HINS4', 'HINS5', 'HINS6']] == 1)  # Medicare, Medicaid, Tricare/VA, Other public
    private = any(row[['HINS1', 'HINS2']] == 1)  # Employer-based, Direct-purchase
    if public and private:
        return 'Both public and private'
    elif public:
        return 'Public only'
    elif private:
        return 'Private only'
    else:
        return 'Not insured'

data['insurance_status'] = data.apply(get_insurance_status, axis=1)
status_percentages = (data['insurance_status'].value_counts(normalize=True) * 100).sort_values()

# Plot insurance status distribution
plt.figure(figsize=(10, 6))
status_percentages.plot(kind='barh', color=sns.color_palette('pastel', 4))
plt.title('Public vs. Private Insurance Status (% of Total)')
plt.xlabel('Percentage (%)')
plt.ylabel('Insurance Status')
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_insurance_status.png", dpi=200, bbox_inches='tight')
plt.close()

# Print results
print("Top 10 Healthcare Coverage Combinations (% of Total):\n", top_10_combinations)
print("\nPublic vs. Private Insurance Status (% of Total):\n", status_percentages)
