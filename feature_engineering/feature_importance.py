"""
Feature Importance Analysis for 01_fe_dataset.csv
Calculates and visualizes feature importance using Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
df = pd.read_csv('output/01_fe_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few columns: {df.columns[:10].tolist()}")
print(f"\nTarget variable (public_coverage_only) distribution:")
print(df['public_coverage_only'].value_counts())

# Prepare features and target
X = df.drop('public_coverage_only', axis=1)
y = df['public_coverage_only']

print(f"\nFeatures shape: {X.shape}")
print(f"Number of features: {len(X.columns)}")

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

# Convert categorical columns to numeric
for col in categorical_cols:
    X[col] = pd.Categorical(X[col]).codes

# Handle any missing values
X = X.fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest model...")
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_importance.head(20).to_string(index=False))

# Save to file
output_file = 'output/feature_importance.csv'
feature_importance.to_csv(output_file, index=False)
print(f"\n✅ Full feature importance saved to: {output_file}")

# Visualize top 20 features
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save plot
plot_file = 'output/feature_importance.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✅ Feature importance plot saved to: {plot_file}")
plt.show()

# Group importance by feature type
print("\n" + "="*60)
print("FEATURE IMPORTANCE BY CATEGORY")
print("="*60)

def categorize_feature(feature_name):
    if feature_name.startswith('OCCP_'):
        return 'Occupation'
    elif feature_name.startswith('INDP_'):
        return 'Industry'
    elif feature_name in ['AGEP', 'age_group']:
        return 'Age'
    elif feature_name in ['SEX']:
        return 'Gender'
    elif feature_name in ['MAR', 'is_married']:
        return 'Marital Status'
    elif feature_name in ['SCHL', 'education_level', 'is_enrolled']:
        return 'Education'
    elif feature_name in ['ESR', 'employment_category', 'is_employed', 'COW']:
        return 'Employment'
    elif feature_name in ['WAGP', 'wage_group', 'has_wages']:
        return 'Wages'
    else:
        return 'Other'

feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)

print(category_importance.to_string())

# Visualize category importance
plt.figure(figsize=(10, 6))
category_importance.plot(kind='bar')
plt.xlabel('Feature Category')
plt.ylabel('Total Importance')
plt.title('Feature Importance by Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/feature_importance_by_category.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Category importance plot saved to: output/feature_importance_by_category.png")
plt.show()

# Model performance
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy:  {test_score:.4f}")
