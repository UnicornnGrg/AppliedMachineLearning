"""
PCA Dimensionality Reduction for Realistic Features

Purpose:
    Apply Principal Component Analysis (PCA) to the realistic (combined) selected features
    to reduce dimensionality while retaining 95% of the variance.

Input:
    - feature_engineering/output/03_fe_realistic_encoded.csv

Output:
    - PCA-transformed dataset
    - PCA model and scaler
    - Analysis reports and visualizations

Author: GitHub Copilot
Date: November 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import fe_utils
import os

warnings.filterwarnings('ignore')

# Set visualization style
fe_utils.setup_environment()

def save_pca_outputs(df_pca, loadings_df, pca, output_dir, prefix):
    """Save PCA datasets and reports."""
    print("\n" + "=" * 60)
    print("SAVING PCA OUTPUTS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PCA dataset
    pca_path = os.path.join(output_dir, f'{prefix}.csv')
    df_pca.to_csv(pca_path, index=False)
    print(f"Saved PCA dataset: {pca_path}")
    
    # Save loadings
    loadings_path = os.path.join(output_dir, f'{prefix}_loadings.csv')
    loadings_df.to_csv(loadings_path)
    print(f"Saved loadings: {loadings_path}")
    
    # Save variance info
    variance_df = pd.DataFrame({
        'Component': range(1, len(pca.explained_variance_ratio_) + 1),
        'Explained_Variance': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_path = os.path.join(output_dir, f'{prefix}_variance.csv')
    variance_df.to_csv(variance_path, index=False)
    print(f"Saved variance info: {variance_path}")
    
    # Generate Report
    report_path = os.path.join(output_dir, f'{prefix}_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PCA ANALYSIS REPORT: {prefix}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DIMENSIONALITY REDUCTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original Dimensions: {loadings_df.shape[0]}\n")
        f.write(f"PCA Dimensions: {df_pca.shape[1] - 1}\n") # -1 for target
        f.write(f"Reduction Ratio: {(1 - (df_pca.shape[1]-1)/loadings_df.shape[0]):.2%}\n")
        f.write(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.4%}\n\n")
        
        f.write("TOP COMPONENTS EXPLAINED VARIANCE\n")
        f.write("-" * 80 + "\n")
        for i in range(min(10, len(pca.explained_variance_ratio_))):
            f.write(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.4%} (Cumulative: {np.cumsum(pca.explained_variance_ratio_)[i]:.4%})\n")
        f.write("\n")
        
        f.write("COMPONENT INTERPRETATION (Top 5 Features by Absolute Loading)\n")
        f.write("-" * 80 + "\n")
        for i in range(min(5, loadings_df.shape[1])):
            pc_col = f'PC{i+1}'
            f.write(f"\n{pc_col}:\n")
            # Get top features
            top_features = loadings_df[pc_col].abs().sort_values(ascending=False).head(5)
            for feat, val in top_features.items():
                loading = loadings_df.loc[feat, pc_col]
                f.write(f"  - {feat}: {loading:.4f}\n")
                
        f.write("\n")
        f.write("CORRELATION WITH TARGET\n")
        f.write("-" * 80 + "\n")
        if 'medicaid_only' in df_pca.columns:
            corrs = df_pca.corrwith(df_pca['medicaid_only']).drop('medicaid_only').sort_values(ascending=False, key=abs)
            for pc, corr in corrs.head(10).items():
                f.write(f"  {pc}: {corr:.4f}\n")
                
    print(f"Saved report: {report_path}")

def main():
    print("\n" + "=" * 80)
    print("PCA DIMENSIONALITY REDUCTION - REALISTIC FEATURES")
    print("=" * 80)
    
    # Configuration
    input_file = '03 feature_engineering/output/03_fe_realistic_encoded.csv'
    output_dir = '03 feature_engineering/output'
    prefix = '06_fe_pca_realistic'
    
    # 1. Load Data
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        # Fallback to absolute path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, 'output', '03_fe_realistic_encoded.csv')
        
    df_encoded = pd.read_csv(input_file)
    print(f"Shape: {df_encoded.shape}")
    
    # 2. Preprocess
    # Verify numeric
    if df_encoded.select_dtypes(exclude=[np.number]).shape[1] > 0:
        raise ValueError("Non-numeric columns remain in encoded dataset!")
        
    # Handle missing values (impute median for any remaining NaNs)
    df_encoded = df_encoded.fillna(df_encoded.median())
    
    # 3. Apply PCA
    df_pca, pca, scaler, feature_names = fe_utils.apply_pca(df_encoded, n_components=0.95)
    
    # 4. Analyze Loadings
    loadings_df = fe_utils.analyze_pca_loadings(pca, feature_names)
    
    # 5. Save Outputs
    fe_utils.save_pca_models(pca, scaler, output_dir, prefix)
    save_pca_outputs(df_pca, loadings_df, pca, output_dir, prefix)
    
    # 6. Visualizations
    fe_utils.plot_pca_scree(pca, os.path.join(output_dir, f'{prefix}_scree.png'))
    fe_utils.plot_pca_cumulative_variance(pca, os.path.join(output_dir, f'{prefix}_cumulative_variance.png'))
    fe_utils.plot_pca_loadings_heatmap(loadings_df, os.path.join(output_dir, f'{prefix}_loadings_heatmap.png'))
    
    # Correlation bar chart
    if 'medicaid_only' in df_pca.columns:
        corrs = df_pca.corrwith(df_pca['medicaid_only']).drop('medicaid_only').abs().sort_values(ascending=False).head(20)
        fe_utils.plot_feature_ranking(
            corrs.index, 
            corrs.values, 
            'Absolute Correlation with Target', 
            os.path.join(output_dir, f'{prefix}_correlation.png'),
            color='purple'
        )
    
    print("\n" + "=" * 80)
    print("PCA ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
