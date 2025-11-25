import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the output directory exists
os.makedirs("01 data_understanding/output", exist_ok=True)

# Load the data
data = pd.read_csv("00 data/processed/psam_p38_cleaned.csv")

# Filter to ages 18-65
data = data[(data['AGEP'] >= 18) & (data['AGEP'] <= 65)]

# Create target variable
data['medicaid_only'] = ((data['HINS4'] == 1) & (data['PRIVCOV'] == 2)).astype(int)

# Print target distribution and class imbalance ratio
target_dist = data['medicaid_only'].value_counts(normalize=True)
print("Target Distribution:\n", target_dist)
print("Class Imbalance Ratio: {:.2f}".format(target_dist[0] / target_dist[1]))

# Drop unnecessary columns
cols_to_drop = [col for col in data.columns if col.startswith(('PWGTP', 'F'))]
cols_to_drop.extend(['HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7', 'PUBCOV', 'PRIVCOV', 'HICOV', 'NAICSP', 'SOCP'])
data = data.drop(columns=cols_to_drop, errors='ignore')

# Summary statistics for all features
print("\nSummary Statistics for All Features:")
print(data.describe(include='all'))

# Function to save plots
def save_plot(plot_func, filename, **kwargs):
    plt.figure(figsize=(12, 8))
    plot_func(**kwargs)
    plt.tight_layout()
    plt.savefig(f"01 data_understanding/output/da_{filename}.png", dpi=200, bbox_inches='tight')
    plt.close()

# Target distribution plot
save_plot(
    sns.countplot,
    "target_distribution",
    data=data, x='medicaid_only'
)

# Numerical features: distributions and boxplots
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('medicaid_only')

# Histograms for all numerical features
for col in num_cols:
    save_plot(
        sns.histplot,
        f"dist_{col}",
        data=data, x=col, kde=True, bins=30
    )

# Boxplots for all numerical features by target
for col in num_cols:
    save_plot(
        sns.boxplot,
        f"box_{col}_by_target",
        data=data, x='medicaid_only', y=col
    )

# Correlation matrix (numerical features only)
plt.figure(figsize=(16, 12))
corr = data[num_cols + ['medicaid_only']].corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 8})
plt.title("Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_correlation_matrix.png", dpi=200, bbox_inches='tight')
plt.close()

# Missing values analysis
missing = data.isnull().sum().sort_values(ascending=False)
missing_percent = (data.isnull().sum() / len(data)) * 100
missing_info = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
missing_info = missing_info[missing_info['Missing Values'] > 0]
print("\nMissing Values Summary:\n", missing_info)

# Missing values heatmap (for a subset of columns, if too many)
if len(data.columns) <= 20:
    save_plot(
        sns.heatmap,
        "missing_values_heatmap",
        data=data.isnull(), cbar=False, cmap='viridis'
    )

# Pairplot for a sample of numerical features (if not too many)
if len(num_cols) <= 10:
    sns.pairplot(data, vars=num_cols, hue='medicaid_only', diag_kind='kde')
    plt.savefig("01 data_understanding/output/da_pairplot.png", dpi=200, bbox_inches='tight')
    plt.close()

# Categorical features: count plots (for top 20 categories)
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    if data[col].nunique() <= 20:
        save_plot(
            sns.countplot,
            f"count_{col}",
            data=data, x=col, hue='medicaid_only'
        )
        plt.xticks(rotation=45)

# Outlier detection: boxplots for all numerical features
plt.figure(figsize=(16, 12))
data[num_cols].boxplot()
plt.title("Outlier Detection (Numerical Features)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("01 data_understanding/output/da_outliers_boxplot.png", dpi=200, bbox_inches='tight')
plt.close()
