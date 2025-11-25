"""
Realistic Feature Selection for Medicaid-Only Insurance Prediction

Purpose:
    Feature selection based on realistic business constraints and interpretability.
    Focuses on a concise set of high-value features that are easy to collect and explain.

Target:
    Predict Medicaid-only individuals (potential market for private insurance upsell)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import fe_utils

warnings.filterwarnings('ignore')

# Set visualization style
fe_utils.setup_environment()


def engineer_income_affordability_features(df):
    """
    GROUP 1: Income & Affordability Features
    """
    print("\n" + "=" * 60)
    print("GROUP 1: INCOME & AFFORDABILITY FEATURES")
    print("=" * 60)
    
    features = []
    
    # Engineer: income_group
    if 'WAGP' in df.columns:
        df['income_group'] = pd.cut(df['WAGP'], 
                                     bins=[0, 15000, 30000, 50000, 75000, np.inf],
                                     labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        features.append('income_group')
        print("Created income_group: Very_Low (<15k), Low (15-30k), Medium (30-50k), High (50-75k), Very_High (>75k)")
    
    # Engineer: poverty_band
    if 'POVPIP' in df.columns:
        df['poverty_band'] = pd.cut(df['POVPIP'],
                                     bins=[0, 100, 138, 200, 400, np.inf],
                                     labels=['<100%', '100-138%', '138-200%', '200-400%', '>400%'])
        features.append('poverty_band')
        print("Created poverty_band: <100%, 100-138%, 138-200%, 200-400%, >400%")
    
    # Engineer: has_wage_income
    if 'WAGP' in df.columns:
        df['has_wage_income'] = (df['WAGP'] > 0).astype(int)
        features.append('has_wage_income')
        print("Created has_wage_income: Binary (WAGP > 0)")
    
    # Engineer: has_self_employment
    if 'SEMP' in df.columns:
        df['has_self_employment'] = (df['SEMP'] > 0).astype(int)
        features.append('has_self_employment')
        print("Created has_self_employment: Binary (SEMP > 0)")
    
    print(f"\nTotal Income & Affordability features: {len(features)}")
    return df, features


def engineer_employment_quality_features(df):
    """
    GROUP 2: Employment Quality Features
    """
    print("\n" + "=" * 60)
    print("GROUP 2: EMPLOYMENT QUALITY FEATURES")
    print("=" * 60)
    
    features = []
    
    # Engineer: employment_status_simple
    if 'ESR' in df.columns:
        esr_mapping = {
            1: 'Employed',      # Civilian employed, at work
            2: 'Employed',      # Civilian employed, with job but not at work
            3: 'Unemployed',    # Unemployed
            4: 'Not_in_labor_force',  # Armed forces, at work
            5: 'Not_in_labor_force',  # Armed forces, with job but not at work
            6: 'Not_in_labor_force'   # Not in labor force
        }
        df['employment_status_simple'] = df['ESR'].map(esr_mapping).fillna('Not_in_labor_force')
        features.append('employment_status_simple')
        print("Created employment_status_simple: Employed, Unemployed, Not_in_labor_force")
    
    # Engineer: employer_type
    if 'COW' in df.columns:
        cow_mapping = {
            1: 'Private',         # Private for-profit
            2: 'Private',         # Private non-profit
            3: 'Government',      # Local government
            4: 'Government',      # State government
            5: 'Government',      # Federal government
            6: 'Self_employed',   # Self-employed incorporated
            7: 'Self_employed',   # Self-employed not incorporated
            8: 'Unpaid',          # Working without pay
            9: 'Unpaid'           # Unemployed
        }
        df['employer_type'] = df['COW'].map(cow_mapping).fillna('Unknown')
        features.append('employer_type')
        print("Created employer_type: Private, Government, Self_employed, Unpaid")
    
    print(f"\nTotal Employment Quality features: {len(features)}")
    return df, features


def engineer_demographics_life_stage_features(df):
    """
    GROUP 3: Demographics & Life Stage Features
    """
    print("\n" + "=" * 60)
    print("GROUP 3: DEMOGRAPHICS & LIFE STAGE FEATURES")
    print("=" * 60)
    
    features = []
    
    # Raw features
    if 'AGEP' in df.columns:
        features.append('AGEP')
    if 'SEX' in df.columns:
        features.append('SEX')
    if 'MAR' in df.columns:
        features.append('MAR')
    
    # Engineer: age_group
    if 'AGEP' in df.columns:
        df['age_group'] = pd.cut(df['AGEP'],
                                 bins=[17, 25, 35, 45, 55, 65],
                                 labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
        features.append('age_group')
        print("Created age_group: 18-25, 26-35, 36-45, 46-55, 56-65")
    
    # Engineer: marital_status_simple
    if 'MAR' in df.columns:
        mar_mapping = {
            1: 'Married',              # Married
            2: 'Divorced_Separated',   # Widowed
            3: 'Divorced_Separated',   # Divorced
            4: 'Divorced_Separated',   # Separated
            5: 'Never_married'         # Never married
        }
        df['marital_status_simple'] = df['MAR'].map(mar_mapping).fillna('Unknown')
        features.append('marital_status_simple')
        print("Created marital_status_simple: Married, Divorced_Separated, Widowed, Never_married")
    
    print(f"\nTotal Demographics & Life Stage features: {len(features)}")
    return df, features


def engineer_education_features(df):
    """
    GROUP 4: Education Features
    """
    print("\n" + "=" * 60)
    print("GROUP 4: EDUCATION FEATURES")
    print("=" * 60)
    
    features = []
    
    # Engineer: education_level
    if 'SCHL' in df.columns:
        def map_education(schl):
            if pd.isna(schl) or schl == 0:
                return 'Unknown'
            elif schl < 16:
                return 'Less_than_HS'
            elif schl <= 17:
                return 'HS_diploma'
            elif schl <= 20:
                return 'Some_college'
            elif schl == 21:
                return 'Bachelors'
            else:
                return 'Graduate'
        
        df['education_level'] = df['SCHL'].apply(map_education)
        features.append('education_level')
        print("Created education_level: Less_than_HS, HS_diploma, Some_college, Bachelors, Graduate")
    
    # Engineer: currently_enrolled
    if 'SCH' in df.columns:
        df['currently_enrolled'] = df['SCH'].isin([2, 3]).astype(int)
        features.append('currently_enrolled')
        print("Created currently_enrolled: Binary (SCH in [2, 3])")
    
    print(f"\nTotal Education features: {len(features)}")
    return df, features


def engineer_health_disability_features(df):
    """
    GROUP 5: Health & Disability Features
    """
    print("\n" + "=" * 60)
    print("GROUP 5: HEALTH & DISABILITY FEATURES")
    print("=" * 60)
    
    features = []
    # No features for this group in realistic scenario
    
    print(f"\nTotal Health & Disability features: {len(features)}")
    return df, features


def add_geography_features(df):
    """
    GROUP 6: Geography Features
    """
    print("\n" + "=" * 60)
    print("GROUP 6: GEOGRAPHY FEATURES")
    print("=" * 60)
    
    features = []
    
    geo_cols = ['STATE', 'PUMA', 'REGION', 'DIVISION']
    for col in geo_cols:
        if col in df.columns:
            features.append(col)
            print(f"Included {col}")
    
    print(f"\nTotal Geography features: {len(features)}")
    return df, features


def select_realistic_features(df):
    """
    Combine all realistic feature groups into final feature set.
    """
    print("\n" + "=" * 60)
    print("BUILDING FINAL REALISTIC FEATURE SET")
    print("=" * 60)
    
    # Engineer features by group
    df, income_features = engineer_income_affordability_features(df)
    df, employment_features = engineer_employment_quality_features(df)
    df, demographics_features = engineer_demographics_life_stage_features(df)
    df, education_features = engineer_education_features(df)
    df, health_features = engineer_health_disability_features(df)
    df, geography_features = add_geography_features(df)
    
    # Combine all features
    all_realistic_features = (income_features + employment_features + 
                            demographics_features + education_features + 
                            health_features + geography_features)
    
    # Add target
    all_realistic_features.append('medicaid_only')
    
    # Select only these columns
    df_realistic = df[all_realistic_features].copy()
    
    print(f"\nFinal realistic feature set: {len(all_realistic_features) - 1} features + 1 target")
    print(f"Final dataset shape: {df_realistic.shape}")
    
    # Create feature metadata
    feature_groups = {}
    for feat in income_features:
        feature_groups[feat] = 'Income_Affordability'
    for feat in employment_features:
        feature_groups[feat] = 'Employment_Quality'
    for feat in demographics_features:
        feature_groups[feat] = 'Demographics_Life_Stage'
    for feat in education_features:
        feature_groups[feat] = 'Education'
    for feat in health_features:
        feature_groups[feat] = 'Health_Disability'
    for feat in geography_features:
        feature_groups[feat] = 'Geography'
        
    return df_realistic, feature_groups


def identify_final_categorical_features(df):
    """
    Identify all categorical features in the final realistic dataset.
    """
    categorical_features = []
    
    for col in df.columns:
        if col == 'medicaid_only':
            continue
        
        # Check if categorical
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)
    
    return categorical_features


def create_encoded_dataset(df, categorical_features):
    """
    Create dummy-encoded version using shared utility.
    """
    df_encoded = fe_utils.create_encoded_dataset(df, categorical_features)
    
    # Calculate dummy breakdown for report
    dummy_breakdown = {}
    for cat_feat in categorical_features:
        dummy_cols = [col for col in df_encoded.columns if col.startswith(f"{cat_feat}_")]
        dummy_breakdown[cat_feat] = len(dummy_cols)
        
    return df_encoded, dummy_breakdown


def calculate_feature_correlations(df, feature_groups):
    """
    Calculate correlation with target for each feature and feature group.
    """
    print("\n" + "=" * 60)
    print("CALCULATING FEATURE CORRELATIONS WITH TARGET")
    print("=" * 60)
    
    # Use shared utility
    correlations = fe_utils.calculate_target_correlation(df)
    
    # Calculate average correlation by group
    group_correlations = {}
    for group in set(feature_groups.values()):
        group_features = [feat for feat, grp in feature_groups.items() if grp == group]
        group_corrs = [correlations[feat] for feat in group_features if feat in correlations]
        if group_corrs:
            group_correlations[group] = np.mean(np.abs(group_corrs))
    
    print(f"\nAverage absolute correlation by realistic group:")
    for group, corr in sorted(group_correlations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {corr:.4f}")
    
    return correlations, group_correlations


def save_outputs(df_categorical, df_encoded, feature_groups, correlations, 
                group_correlations, dummy_breakdown, output_dir):
    """
    Save all output files.
    """
    # Save datasets using shared util
    fe_utils.save_datasets(df_categorical, df_encoded, output_dir, '03_fe_realistic')
    
    output_dir = Path(output_dir)
    
    # 3. Feature metadata
    metadata = []
    for feat in df_categorical.columns:
        if feat == 'medicaid_only':
            continue
        
        metadata.append({
            'feature_name': feat,
            'business_group': feature_groups.get(feat, 'Unknown'),
            'data_type': str(df_categorical[feat].dtype),
            'unique_values': df_categorical[feat].nunique(),
            'correlation_with_target': correlations.get(feat, np.nan)
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / '03_fe_realistic_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSaved feature metadata: {metadata_path}")
    
    # 4. Realistic report
    report_path = output_dir / '03_fe_realistic_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REALISTIC FEATURE SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total features selected: {len(df_categorical.columns) - 1}\n")
        f.write(f"Target variable: medicaid_only\n")
        f.write(f"Dataset size: {len(df_categorical):,} records\n\n")
        
        f.write("FEATURE COUNT BY GROUP\n")
        f.write("-" * 80 + "\n")
        group_counts = metadata_df['business_group'].value_counts()
        for group, count in group_counts.items():
            f.write(f"{group}: {count} features\n")
        f.write("\n")
        
        f.write("ENGINEERED FEATURES\n")
        f.write("-" * 80 + "\n")
        engineered = [
            ('income_group', 'Income categorization for market segmentation'),
            ('poverty_band', 'Federal Poverty Level bands (Medicaid eligibility)'),
            ('has_wage_income', 'Binary indicator for wage income'),
            ('has_self_employment', 'Binary indicator for self-employment income'),
            ('employment_status_simple', 'Simplified employment status'),
            ('employer_type', 'Simplified employer type'),
            ('age_group', 'Life stage segmentation'),
            ('marital_status_simple', 'Simplified marital status'),
            ('education_level', 'Education attainment level'),
            ('currently_enrolled', 'Student enrollment status')
        ]
        for feat, desc in engineered:
            if feat in df_categorical.columns:
                f.write(f"{feat}: {desc}\n")
        f.write("\n")
        
        f.write("CORRELATION WITH TARGET BY FEATURE\n")
        f.write("-" * 80 + "\n")
        f.write("Top 20 features by absolute correlation:\n\n")
        top_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        for feat, corr in top_corrs:
            f.write(f"  {feat:40s} {corr:7.4f}\n")
        f.write("\n")
        
        f.write("CORRELATION WITH TARGET BY GROUP\n")
        f.write("-" * 80 + "\n")
        for group, corr in sorted(group_correlations.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{group:30s} {corr:.4f}\n")
        f.write("\n")
        
        f.write("DUMMY ENCODING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total categorical features encoded: {len(dummy_breakdown)}\n")
        f.write(f"Total dummy columns created: {sum(dummy_breakdown.values())}\n")
        f.write(f"Final encoded feature count: {len(df_encoded.columns) - 1}\n\n")
        f.write("Top 10 features by dummy expansion:\n")
        for feat, count in sorted(dummy_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  {feat:30s} {count:3d} dummies\n")
    
    print(f"\nSaved realistic report: {report_path}")


def create_visualizations(df_categorical, df_encoded, feature_groups, 
                         correlations, group_correlations, output_dir):
    """
    Create and save all visualizations.
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    
    # 1. Feature count by group
    plt.figure(figsize=(12, 6))
    group_counts = pd.Series(feature_groups).value_counts().sort_values(ascending=True)
    group_counts.plot(kind='barh', color='steelblue')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.title('Feature Count by Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '03_fe_realistic_feature_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 03_fe_realistic_feature_counts.png")
    
    # 2. Average correlation by group
    plt.figure(figsize=(12, 6))
    group_corr_series = pd.Series(group_correlations).sort_values(ascending=True)
    group_corr_series.plot(kind='barh', color='coral')
    plt.xlabel('Average Absolute Correlation with Target', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.title('Average Correlation with Target by Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '03_fe_realistic_group_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 03_fe_realistic_group_correlations.png")
    
    # 3. Correlation heatmap (top features by group)
    numeric_features = [col for col in df_categorical.columns 
                       if df_categorical[col].dtype in ['int64', 'float64'] and col != 'medicaid_only']
    
    if len(numeric_features) > 0:
        # Select top 3 features per group by absolute correlation
        features_to_plot = []
        for group in set(feature_groups.values()):
            group_features = [feat for feat, grp in feature_groups.items() 
                            if grp == group and feat in correlations]
            group_top = sorted(group_features, key=lambda x: abs(correlations[x]), reverse=True)[:3]
            features_to_plot.extend(group_top)
        
        fe_utils.plot_correlation_heatmap(
            df_categorical, 
            output_dir / '03_fe_realistic_correlation_heatmap.png',
            top_features=features_to_plot
        )
    
    # 4. Target rate by key categorical features
    categorical_to_plot = []
    
    for feat in ['employment_status_simple', 'education_level', 'poverty_band', 'age_group']:
        if feat in df_categorical.columns:
            categorical_to_plot.append(feat)
    
    if categorical_to_plot:
        n_plots = len(categorical_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, feat in enumerate(categorical_to_plot):
            if idx < len(axes):
                target_rate = df_categorical.groupby(feat)['medicaid_only'].mean().sort_values(ascending=False)
                
                ax = axes[idx]
                target_rate.plot(kind='bar', ax=ax, color='teal')
                ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=11)
                ax.set_ylabel('Medicaid-Only Rate', fontsize=11)
                ax.set_title(f'Target Rate by {feat.replace("_", " ").title()}', 
                           fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(categorical_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / '03_fe_realistic_target_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 03_fe_realistic_target_distributions.png")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 80)
    print("REALISTIC FEATURE SELECTION")
    print("PUMS Census Data - Medicaid-Only Insurance Prediction")
    print("=" * 80)
    
    # File paths
    input_file = '00 data/processed/psam_p38_cleaned.csv'
    output_dir = '03 feature_engineering/output'
    
    # 1. Load and prepare data
    df = fe_utils.load_and_prepare_data(input_file)
    
    # 2. Run standard preprocessing pipeline (Identify, Group, Missing)
    df, categorical_cols, numeric_cols = fe_utils.run_preprocessing_pipeline(df)
    
    # 3. Select realistic features
    df_realistic, feature_groups = select_realistic_features(df)
    
    # 4. Identify categorical features in final dataset
    categorical_features = identify_final_categorical_features(df_realistic)
    
    # 5. Create encoded dataset
    df_encoded, dummy_breakdown = create_encoded_dataset(df_realistic, categorical_features)
    
    # 6. Calculate correlations
    correlations, group_correlations = calculate_feature_correlations(df_realistic, feature_groups)
    
    # 7. Save outputs
    save_outputs(df_realistic, df_encoded, feature_groups, correlations, 
                group_correlations, dummy_breakdown, output_dir)
    
    # 8. Create visualizations
    create_visualizations(df_realistic, df_encoded, feature_groups, 
                         correlations, group_correlations, output_dir)
    
    # 9. Print verification
    fe_utils.print_verification(df_encoded)
    
    print("\n" + "=" * 80)
    print("REALISTIC FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - 03_fe_realistic_categorical.csv")
    print("  - 03_fe_realistic_encoded.csv")
    print("  - 03_fe_realistic_metadata.csv")
    print("  - 03_fe_realistic_report.txt")
    print("  - 03_fe_realistic_*.png (visualizations)")


if __name__ == "__main__":
    main()
