"""
Feature Engineering - Experiment 01
====================================
Target Variable: People with ONLY public healthcare coverage
Features: Demographics, Education, Employment

Healthcare Coverage Variables (PUMS Data):
- HINS1: Insurance through current or former employer or union
- HINS2: Insurance purchased directly from insurance company
- HINS3: Medicare (65+ or with certain disabilities)
- HINS4: Medicaid, Medical Assistance, or government assistance plan
- HINS5: TRICARE or other military health care
- HINS6: VA health care
- HINS7: Indian Health Service
- PUBCOV: Public coverage (1 = Yes, 2 = No)
- PRIVCOV: Private coverage

Target: Binary variable (0/1) where:
       1 = PUBCOV = 1 (has public coverage) AND PRIVCOV = 2 (no private coverage)
       0 = All other cases
       This identifies people with ONLY public healthcare coverage.

Features Used:
1. Demographics: AGEP (age), SEX (sex), MAR (marital status)
2. Education: SCHL (educational attainment), SCH (school enrollment)
3. Employment: OCCP (occupation - grouped by 2-digit code), INDP (industry - grouped by 2-digit code), 
               ESR (employment status), WAGP (salary)

Note: Occupation (OCCP) and Industry (INDP) are 4-digit codes grouped to 2-digit codes,
      then converted to dummy variables to reduce dimensionality.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
INPUT_FILE = PROCESSED_DIR / 'psam_p38_cleaned.csv'
OUTPUT_DIR = PROJECT_ROOT / 'feature_engineering' / 'output'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the cleaned dataset and filter by age."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Filter to keep only individuals aged 18-65
    original_count = len(df)
    df = df[(df['AGEP'] >= 18) & (df['AGEP'] <= 65)].copy()
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    print(f"Filtered to ages 18-65: {filtered_count} records (removed {removed_count} records)")
    
    return df

def create_target_variable(df):
    """
    Create binary target variable: People with ONLY public healthcare coverage.
    
    Creates a new column 'public_coverage_only' where:
    - 1 = Has public coverage (PUBCOV=1) AND no private coverage (PRIVCOV=2)
    - 0 = All other cases (has private coverage, no coverage, or both)
    
    This is the dependent variable for modeling.
    """
    print("\nCreating target variable...")
    
    # PUBCOV: 1 = Yes (has public coverage), 2 = No
    # PRIVCOV: 1 = Yes (has private coverage), 2 = No
    # Create new binary target variable
    df['public_coverage_only'] = ((df['PUBCOV'] == 1) & (df['PRIVCOV'] == 2)).astype(int)
    
    # Print distribution
    print(f"\nTarget Variable: 'public_coverage_only'")
    print(f"  Value 1 (Only Public Coverage): {df['public_coverage_only'].sum()} ({df['public_coverage_only'].mean()*100:.2f}%)")
    print(f"  Value 0 (Other Coverage): {(df['public_coverage_only']==0).sum()} ({(1-df['public_coverage_only'].mean())*100:.2f}%)")
    
    return df

def engineer_demographic_features(df):
    """
    Engineer demographic features.
    
    Features:
    - AGEP: Age (continuous)
    - age_group: Age grouped into categories
    - SEX: Sex (1=Male, 2=Female)
    - MAR: Marital status (1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married)
    - is_married: Binary indicator for married status
    """
    print("\nEngineering demographic features...")
    
    # Age groups (for ages 18-65 only)
    df['age_group'] = pd.cut(df['AGEP'], 
                              bins=[17, 25, 35, 45, 55, 65],
                              labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    
    # Binary married indicator (MAR: 1=Married)
    df['is_married'] = (df['MAR'] == 1).astype(int)
    
    # Sex is already coded (1=Male, 2=Female), keep as is
    
    print(f"  Age range: {df['AGEP'].min()} - {df['AGEP'].max()}")
    print(f"  Age groups distribution:")
    print(df['age_group'].value_counts().sort_index())
    print(f"\n  Sex distribution:")
    print(df['SEX'].value_counts())
    print(f"\n  Marital status distribution:")
    print(df['MAR'].value_counts())
    
    return df

def engineer_education_features(df):
    """
    Engineer education features.
    
    Features:
    - SCHL: Educational attainment (1-24, with specific levels)
    - education_level: Simplified education categories
    - SCH: School enrollment (1=No, 2=Yes public, 3=Yes private)
    - is_enrolled: Binary indicator for school enrollment
    """
    print("\nEngineering education features...")
    
    # Simplify education levels
    # SCHL codes (approximate): 1-15=No HS diploma, 16-17=HS/GED, 18-20=Some college, 21=Bachelor's, 22-24=Graduate
    def categorize_education(schl):
        if pd.isna(schl):
            return 'Unknown'
        elif schl < 16:
            return 'No HS Diploma'
        elif schl <= 17:
            return 'HS or GED'
        elif schl <= 20:
            return 'Some College'
        elif schl == 21:
            return 'Bachelor\'s'
        else:
            return 'Graduate Degree'
    
    df['education_level'] = df['SCHL'].apply(categorize_education)
    
    # School enrollment indicator (SCH: 1=No, 2 or 3=Yes)
    df['is_enrolled'] = (df['SCH'].isin([2, 3])).astype(int)
    
    print(f"  Education level distribution:")
    print(df['education_level'].value_counts())
    print(f"\n  School enrollment:")
    print(f"    Enrolled: {df['is_enrolled'].sum()} ({df['is_enrolled'].mean()*100:.2f}%)")
    print(f"    Not enrolled: {(df['is_enrolled']==0).sum()} ({(1-df['is_enrolled'].mean())*100:.2f}%)")
    
    return df

def engineer_employment_features(df):
    """
    Engineer employment features.
    
    Features:
    - ESR: Employment status (1=Employed, 2=Employed not at work, 3=Unemployed, 4-6=Not in labor force)
    - is_employed: Binary indicator for employment
    - employment_category: Simplified employment status
    - WAGP: Wages/salary (continuous)
    - wage_group: Wage categories
    - has_wages: Binary indicator for having wages
    - OCCP_2digit: Occupation grouped by first 2 digits of 4-digit code
    - INDP_2digit: Industry grouped by first 2 digits of 4-digit code
    - COW: Class of worker (categorical)
    """
    print("\nEngineering employment features...")
    
    # Employment status categories
    def categorize_employment(esr):
        if pd.isna(esr):
            return 'Unknown'
        elif esr in [1, 2]:
            return 'Employed'
        elif esr == 3:
            return 'Unemployed'
        else:
            return 'Not in Labor Force'
    
    df['employment_category'] = df['ESR'].apply(categorize_employment)
    df['is_employed'] = (df['ESR'].isin([1, 2])).astype(int)
    
    # Wage features
    df['has_wages'] = (df['WAGP'] > 0).astype(int)
    
    # Wage groups (only for those with wages)
    df['wage_group'] = pd.cut(df[df['WAGP'] > 0]['WAGP'],
                               bins=[0, 15000, 30000, 50000, 75000, 100000, np.inf],
                               labels=['<15k', '15k-30k', '30k-50k', '50k-75k', '75k-100k', '100k+'])
    
    # Group occupation by first 2 digits of 4-digit code
    # Convert to string, take first 2 characters, handle missing values
    df['OCCP_2digit'] = df['OCCP'].apply(lambda x: str(int(x))[:2] if pd.notna(x) else 'Missing')
    
    # Group industry by first 2 digits of 4-digit code
    df['INDP_2digit'] = df['INDP'].apply(lambda x: str(int(x))[:2] if pd.notna(x) else 'Missing')
    
    print(f"  Employment status distribution:")
    print(df['employment_category'].value_counts())
    print(f"\n  Employed: {df['is_employed'].sum()} ({df['is_employed'].mean()*100:.2f}%)")
    
    print(f"\n  Wage statistics (for those with wages > 0):")
    wage_stats = df[df['WAGP'] > 0]['WAGP'].describe()
    print(wage_stats)
    
    print(f"\n  Wage group distribution:")
    print(df['wage_group'].value_counts().sort_index())
    
    print(f"\n  Has wages: {df['has_wages'].sum()} ({df['has_wages'].mean()*100:.2f}%)")
    
    # Print occupation and industry grouping summary
    print(f"\n  Occupation 2-digit groups: {df['OCCP_2digit'].nunique()} unique groups")
    print(f"  Top 10 occupation groups:")
    print(df['OCCP_2digit'].value_counts().head(10))
    
    print(f"\n  Industry 2-digit groups: {df['INDP_2digit'].nunique()} unique groups")
    print(f"  Top 10 industry groups:")
    print(df['INDP_2digit'].value_counts().head(10))
    
    return df

def select_features_for_modeling(df):
    """
    Select and prepare final feature set for modeling.
    
    Returns a dataframe with:
    - Target variable (public_coverage_only)
    - Demographic features (numeric and categorical)
    - Education features (numeric and categorical)
    - Employment features (numeric and categorical with dummy variables for occupation/industry)
    """
    print("\nSelecting features for modeling...")
    
    # Define feature columns (before dummy encoding)
    demographic_features = [
        'AGEP',              # Age (numeric)
        'age_group',         # Age category (categorical)
        'SEX',               # Sex (categorical: 1=Male, 2=Female)
        'MAR',               # Marital status (categorical)
        'is_married'         # Married indicator (binary)
    ]
    
    education_features = [
        'SCHL',              # Education attainment code (ordinal)
        'education_level',   # Education category (categorical)
        # Removed SCH - redundant with is_enrolled
        'is_enrolled'        # Enrolled indicator (binary)
    ]
    
    employment_features = [
        'ESR',               # Employment status code (categorical)
        'employment_category', # Employment category (categorical)
        'is_employed',       # Employed indicator (binary)
        'COW',               # Class of worker (categorical)
        'WAGP',              # Wages (numeric)
        'wage_group',        # Wage category (categorical)
        'has_wages',         # Has wages indicator (binary)
        'OCCP_2digit',       # Occupation 2-digit group (will be dummy encoded)
        'INDP_2digit'        # Industry 2-digit group (will be dummy encoded)
    ]
    
    # Combine all features
    all_features = demographic_features + education_features + employment_features
    
    # Create base modeling dataset
    modeling_df = df[['public_coverage_only'] + all_features].copy()
    
    print(f"\nDataset shape before dummy encoding: {modeling_df.shape}")
    print(f"  Target variable: public_coverage_only")
    print(f"  Demographic features: {len(demographic_features)}")
    print(f"  Education features: {len(education_features)}")
    print(f"  Employment features: {len(employment_features)}")
    print(f"  Total features (before dummy encoding): {len(all_features)}")
    
    # Create dummy variables for occupation and industry 2-digit groups
    print(f"\nCreating dummy variables for occupation and industry...")
    
    # Get dummy variables for occupation (drop first to avoid multicollinearity)
    occp_dummies = pd.get_dummies(modeling_df['OCCP_2digit'], prefix='OCCP', drop_first=True)
    print(f"  Created {len(occp_dummies.columns)} occupation dummy variables")
    
    # Get dummy variables for industry (drop first to avoid multicollinearity)
    indp_dummies = pd.get_dummies(modeling_df['INDP_2digit'], prefix='INDP', drop_first=True)
    print(f"  Created {len(indp_dummies.columns)} industry dummy variables")
    
    # Drop original OCCP_2digit and INDP_2digit columns
    modeling_df = modeling_df.drop(['OCCP_2digit', 'INDP_2digit'], axis=1)
    
    # Add dummy variables to the dataset
    modeling_df = pd.concat([modeling_df, occp_dummies, indp_dummies], axis=1)
    
    print(f"\nFinal dataset shape after dummy encoding: {modeling_df.shape}")
    print(f"  Total features (including dummies): {len(modeling_df.columns) - 1}")
    
    # Check for missing values
    print(f"\nMissing values per feature:")
    missing_counts = modeling_df.isnull().sum()
    missing_pct = (missing_counts / len(modeling_df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts[missing_counts > 0],
        'Missing %': missing_pct[missing_counts > 0]
    })
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print("  No missing values found!")
    
    return modeling_df

def calculate_kpis(modeling_df):
    """
    Calculate Key Performance Indicators and statistics about the dataset.
    
    Includes:
    - Feature correlations with target
    - Class imbalance metrics
    - Feature variance and missing value analysis
    """
    print("\nCalculating KPIs and statistics...")
    
    kpi_output = OUTPUT_DIR / '01_fe_kpis.txt'
    
    with open(kpi_output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Feature Engineering KPIs - Experiment 01\n")
        f.write("="*70 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total samples: {len(modeling_df)}\n")
        f.write(f"Total features: {len(modeling_df.columns) - 1}\n")
        f.write(f"Age range: 18-65 years\n\n")
        
        # Target variable distribution
        f.write("TARGET VARIABLE DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        target_counts = modeling_df['public_coverage_only'].value_counts()
        target_pct = modeling_df['public_coverage_only'].value_counts(normalize=True) * 100
        f.write(f"Class 0 (Other coverage): {target_counts[0]} ({target_pct[0]:.2f}%)\n")
        f.write(f"Class 1 (Public only): {target_counts[1]} ({target_pct[1]:.2f}%)\n")
        
        # Calculate class imbalance ratio
        imbalance_ratio = target_counts[0] / target_counts[1]
        f.write(f"Imbalance ratio: {imbalance_ratio:.2f}:1\n\n")
        
        # Feature correlations with target
        f.write("TOP 20 FEATURES BY CORRELATION WITH TARGET\n")
        f.write("-"*70 + "\n")
        
        # Select numeric and boolean columns for correlation (includes dummy variables)
        numeric_cols = modeling_df.select_dtypes(include=[np.number, 'bool']).columns
        numeric_df = modeling_df[numeric_cols].astype(float)  # Convert bool to float for correlation
        
        # Calculate correlations
        correlations = numeric_df.corr()['public_coverage_only'].drop('public_coverage_only')
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        f.write("Feature Name                                    Correlation\n")
        f.write("-"*70 + "\n")
        for feat, corr_val in correlations.loc[correlations_abs.head(20).index].items():
            f.write(f"{feat[:45]:<45} {corr_val:>8.4f}\n")
        
        f.write("\n")
        
        # Feature variance analysis
        f.write("FEATURE VARIANCE ANALYSIS\n")
        f.write("-"*70 + "\n")
        variances = numeric_df.var().sort_values(ascending=False)
        f.write(f"Features with highest variance (top 10):\n")
        for feat, var_val in variances.head(10).items():
            if feat != 'public_coverage_only':
                f.write(f"  {feat[:40]:<40} {var_val:>12.2f}\n")
        
        f.write(f"\nFeatures with zero or near-zero variance (< 0.01):\n")
        low_var_features = variances[variances < 0.01]
        low_var_features = low_var_features.drop('public_coverage_only', errors='ignore')
        if len(low_var_features) > 0:
            for feat, var_val in low_var_features.items():
                f.write(f"  {feat[:40]:<40} {var_val:>12.6f}\n")
        else:
            f.write("  None found\n")
        
        f.write("\n")
        
        # Missing values analysis
        f.write("MISSING VALUES ANALYSIS\n")
        f.write("-"*70 + "\n")
        missing_counts = modeling_df.isnull().sum()
        missing_pct = (missing_counts / len(modeling_df) * 100)
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            f.write("Features with missing values:\n")
            for feat, count in features_with_missing.items():
                pct = missing_pct[feat]
                f.write(f"  {feat[:40]:<40} {count:>6} ({pct:>5.2f}%)\n")
        else:
            f.write("No missing values found in dataset\n")
        
        f.write("\n")
        
        # Feature type breakdown
        f.write("FEATURE TYPE BREAKDOWN\n")
        f.write("-"*70 + "\n")
        
        # Count different feature types by prefix/pattern
        base_features = [col for col in modeling_df.columns if not col.startswith(('OCCP_', 'INDP_'))]
        occp_dummies = [col for col in modeling_df.columns if col.startswith('OCCP_')]
        indp_dummies = [col for col in modeling_df.columns if col.startswith('INDP_')]
        
        f.write(f"Base features: {len(base_features) - 1}\n")  # -1 for target
        f.write(f"Occupation dummy variables: {len(occp_dummies)}\n")
        f.write(f"Industry dummy variables: {len(indp_dummies)}\n")
        f.write(f"Total features: {len(modeling_df.columns) - 1}\n\n")
        
        # Summary statistics for key numeric features
        f.write("SUMMARY STATISTICS FOR KEY NUMERIC FEATURES\n")
        f.write("-"*70 + "\n")
        
        key_features = ['AGEP', 'WAGP']
        for feat in key_features:
            if feat in modeling_df.columns:
                f.write(f"\n{feat}:\n")
                stats = modeling_df[feat].describe()
                f.write(f"  Mean:   {stats['mean']:>10.2f}\n")
                f.write(f"  Std:    {stats['std']:>10.2f}\n")
                f.write(f"  Min:    {stats['min']:>10.2f}\n")
                f.write(f"  25%:    {stats['25%']:>10.2f}\n")
                f.write(f"  Median: {stats['50%']:>10.2f}\n")
                f.write(f"  75%:    {stats['75%']:>10.2f}\n")
                f.write(f"  Max:    {stats['max']:>10.2f}\n")
    
    print(f"  Calculated and saved KPIs: {kpi_output}")
    
    # Create correlation matrix heatmap (top 30 most correlated features with target)
    print(f"\n  Creating correlation matrix visualization...")
    
    # Get top 30 features by absolute correlation with target
    top_features = correlations_abs.head(30).index.tolist()
    top_features.append('public_coverage_only')  # Add target
    
    # Create correlation matrix for top features
    correlation_matrix_top = numeric_df[top_features].corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix_top, 
                annot=False,  # Don't show values (too many)
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix - Top 30 Features by Target Correlation', 
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save figure
    corr_fig_output = OUTPUT_DIR / '01_fe_correlation_matrix.png'
    plt.savefig(corr_fig_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation matrix figure: {corr_fig_output}")
    
    # Create second correlation matrix showing occupation on Y-axis and industry on X-axis
    print(f"\n  Creating occupation vs industry correlation matrix...")
    
    # Get occupation and industry dummy columns
    occp_cols = sorted([col for col in numeric_df.columns if col.startswith('OCCP_')])
    indp_cols = sorted([col for col in numeric_df.columns if col.startswith('INDP_')])
    
    if len(occp_cols) > 0 and len(indp_cols) > 0:
        # Create a rectangular correlation matrix: occupation rows x industry columns
        # Calculate correlations between occupation and industry features
        occ_ind_corr_data = []
        
        for occp in occp_cols:
            row_corrs = []
            for indp in indp_cols:
                corr_val = numeric_df[occp].corr(numeric_df[indp])
                row_corrs.append(corr_val)
            occ_ind_corr_data.append(row_corrs)
        
        # Create DataFrame for better labeling
        occ_ind_corr_df = pd.DataFrame(occ_ind_corr_data, 
                                        index=occp_cols, 
                                        columns=indp_cols)
        
        # Create figure - rectangular, not square
        fig_width = max(12, len(indp_cols) * 0.4)
        fig_height = max(10, len(occp_cols) * 0.4)
        plt.figure(figsize=(fig_width, fig_height))
        
        sns.heatmap(occ_ind_corr_df,
                    annot=False,
                    cmap='coolwarm',
                    center=0,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)
        
        plt.title(f'Correlation Matrix: Occupation (rows) vs Industry (columns)', 
                  fontsize=16, pad=20, fontweight='bold')
        plt.xlabel(f'Industry (INDP) - {len(indp_cols)} categories', fontsize=12, fontweight='bold')
        plt.ylabel(f'Occupation (OCCP) - {len(occp_cols)} categories', fontsize=12, fontweight='bold')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save figure
        corr_occ_ind_output = OUTPUT_DIR / '01_fe_correlation_matrix_occ_ind.png'
        plt.savefig(corr_occ_ind_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved occupation vs industry correlation matrix: {corr_occ_ind_output}")
        print(f"    (Y-axis: {len(occp_cols)} occupation codes, X-axis: {len(indp_cols)} industry codes)")
    else:
        print(f"  Skipped occupation/industry correlation matrix (insufficient features)")
    
    # Print summary to console
    print(f"\n  Key findings:")
    print(f"    - Class imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"    - Top correlated feature: {correlations_abs.index[0]} ({correlations[correlations_abs.index[0]]:.4f})")
    print(f"    - Features with missing values: {len(features_with_missing)}")
    print(f"    - Low variance features: {len(low_var_features)}")

def save_outputs(modeling_df):
    """Save the modeling dataset."""
    print("\nSaving outputs...")
    
    # Save modeling dataset (target + all features ready for modeling)
    modeling_output = OUTPUT_DIR / '01_fe_dataset.csv'
    modeling_df.to_csv(modeling_output, index=False)
    print(f"  Saved modeling dataset: {modeling_output}")
    print(f"  Shape: {modeling_df.shape} (rows x columns)")
    print(f"  Target: public_coverage_only")
    print(f"  Features: {len(modeling_df.columns) - 1}")
    
    # Calculate KPIs
    calculate_kpis(modeling_df)

def main():
    """Main execution function."""
    print("="*60)
    print("Feature Engineering - Experiment 01")
    print("Target: People with ONLY public healthcare coverage")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create target variable
    df = create_target_variable(df)
    
    # Engineer features
    df = engineer_demographic_features(df)
    df = engineer_education_features(df)
    df = engineer_employment_features(df)
    
    # Select features for modeling
    modeling_df = select_features_for_modeling(df)
    
    # Save outputs
    save_outputs(modeling_df)
    
    print("\n" + "="*60)
    print("Feature engineering complete!")
    print("="*60)

if __name__ == "__main__":
    main()
