"""
Statistical Feature Selection for PUMS Census Data
===================================================
Purpose: Data-driven feature selection using statistical criteria (correlation, 
         variance, multicollinearity, mutual information) to predict Medicaid-only 
         insurance status.

Approach: 
- Filter to ages 18-65
- Create binary target: medicaid_only = (HINS4 == 1) & (PRIVCOV == 2)
- Remove low-information features (low variance, single-category dominance)
- Remove highly correlated features (reduce redundancy)
- Select top features by mutual information (relevance to target)
- Save categorical and encoded versions for modeling

Author: Applied Machine Learning Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import fe_utils

warnings.filterwarnings('ignore')

# Set plot style
fe_utils.setup_environment()





def filter_low_information(df, categorical_cols, numeric_cols, target='medicaid_only'):
    """
    Remove features with low information content.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns
    numeric_cols : list
        List of numeric columns
    target : str
        Target variable name
    
    Returns:
    --------
    df : pd.DataFrame
        Filtered dataframe
    categorical_cols : list
        Updated categorical columns
    numeric_cols : list
        Updated numeric columns
    """
    print("\n" + "=" * 60)
    print("STEP 3A: LOW-INFORMATION FILTERING")
    print("=" * 60)
    
    removed_features = []
    
    # Filter numeric features
    print("\nFiltering numeric features:")
    print("-" * 40)
    numeric_to_remove = []
    
    for col in numeric_cols:
        # Check variance
        std = df[col].std()
        if std < 1e-4:
            print(f"  Removing {col}: near-zero variance (std={std:.6f})")
            numeric_to_remove.append(col)
            removed_features.append((col, 'near-zero variance'))
            continue
        
        # Check concentration (same value in > 99% of rows)
        value_counts = df[col].value_counts()
        if len(value_counts) > 0:
            max_freq = value_counts.iloc[0] / len(df)
            if max_freq > 0.99:
                print(f"  Removing {col}: {max_freq:.2%} concentration in single value")
                numeric_to_remove.append(col)
                removed_features.append((col, f'{max_freq:.2%} concentration'))
                continue
    
    # Filter categorical features
    print("\nFiltering categorical features:")
    print("-" * 40)
    categorical_to_remove = []
    
    for col in categorical_cols:
        # Check unique values
        nunique = df[col].nunique()
        if nunique < 2:
            print(f"  Removing {col}: only {nunique} unique value(s)")
            categorical_to_remove.append(col)
            removed_features.append((col, f'only {nunique} unique values'))
            continue
        
        # Check concentration
        value_counts = df[col].value_counts()
        max_freq = value_counts.iloc[0] / len(df)
        if max_freq > 0.99:
            print(f"  Removing {col}: {max_freq:.2%} concentration in '{value_counts.index[0]}'")
            categorical_to_remove.append(col)
            removed_features.append((col, f'{max_freq:.2%} concentration'))
            continue
    
    # Apply removals
    all_to_remove = numeric_to_remove + categorical_to_remove
    df = df.drop(columns=all_to_remove)
    numeric_cols = [col for col in numeric_cols if col not in numeric_to_remove]
    categorical_cols = [col for col in categorical_cols if col not in categorical_to_remove]
    
    print(f"\n✓ Removed {len(removed_features)} low-information features")
    print(f"  - Numeric: {len(numeric_to_remove)}")
    print(f"  - Categorical: {len(categorical_to_remove)}")
    print(f"Remaining features: {len(numeric_cols) + len(categorical_cols)}")
    
    return df, categorical_cols, numeric_cols, removed_features


def filter_redundant_features(df, categorical_cols, numeric_cols, target='medicaid_only'):
    """
    Remove highly correlated features to reduce redundancy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns
    numeric_cols : list
        List of numeric columns
    target : str
        Target variable name
    
    Returns:
    --------
    df : pd.DataFrame
        Filtered dataframe
    numeric_cols : list
        Updated numeric columns
    redundant_features : list
        List of removed features with reasons
    """
    print("\n" + "=" * 60)
    print("STEP 3B: REDUNDANCY FILTERING (HIGH CORRELATION)")
    print("=" * 60)
    
    redundant_features = []
    
    if len(numeric_cols) > 1:
        print("\nCalculating pairwise correlations for numeric features...")
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > 0.98:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.98)")
        
        # Decide which to remove
        numeric_to_remove = []
        for col1, col2, corr_val in high_corr_pairs:
            if col1 in numeric_to_remove or col2 in numeric_to_remove:
                continue
            
            # Check missing values
            missing1 = df[col1].isna().sum()
            missing2 = df[col2].isna().sum()
            
            if missing1 > missing2:
                remove_col = col1
                keep_col = col2
            elif missing2 > missing1:
                remove_col = col2
                keep_col = col1
            else:
                # If tied, remove first one alphabetically
                remove_col = col1 if col1 < col2 else col2
                keep_col = col2 if col1 < col2 else col1
            
            print(f"  Removing {remove_col}: r={corr_val:.3f} with {keep_col}")
            numeric_to_remove.append(remove_col)
            redundant_features.append((remove_col, f'r={corr_val:.3f} with {keep_col}'))
        
        # Apply removals
        df = df.drop(columns=numeric_to_remove)
        numeric_cols = [col for col in numeric_cols if col not in numeric_to_remove]
        
        print(f"\n✓ Removed {len(numeric_to_remove)} redundant numeric features")
    else:
        print("\nSkipping correlation analysis: < 2 numeric features")
    
    print(f"Remaining features: {len(numeric_cols) + len(categorical_cols)}")
    
    return df, numeric_cols, redundant_features


def select_features_by_mutual_information(df, categorical_cols, numeric_cols, 
                                          target='medicaid_only', top_k=70):
    """
    Select top K features by mutual information score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns
    numeric_cols : list
        List of numeric columns
    target : str
        Target variable name
    top_k : int
        Number of top features to select
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with selected features
    selected_features : list
        List of selected feature names
    mi_scores : pd.DataFrame
        DataFrame with MI scores for all features
    """
    print("\n" + "=" * 60)
    print("STEP 3C: MUTUAL INFORMATION SELECTION")
    print("=" * 60)
    
    print("\nEncoding categorical features for MI calculation...")
    
    # Create a copy for encoding
    df_encoded = df.copy()
    
    # Encode categorical features using label encoding for MI calculation
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le
    
    # Prepare feature matrix
    feature_cols = numeric_cols + categorical_cols
    X = df_encoded[feature_cols]
    y = df_encoded[target]
    
    print(f"\nCalculating mutual information for {len(feature_cols)} features...")
    
    # Calculate MI scores
    # For categorical features, we specify them as discrete
    categorical_mask = [col in categorical_cols for col in feature_cols]
    mi_scores_array = mutual_info_classif(X, y, discrete_features=categorical_mask, 
                                          random_state=42, n_neighbors=5)
    
    # Create DataFrame with scores
    mi_scores = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores_array,
        'feature_type': ['categorical' if col in categorical_cols else 'numeric' 
                        for col in feature_cols]
    }).sort_values('mi_score', ascending=False)
    
    # Also calculate correlation for numeric features
    correlations = fe_utils.calculate_target_correlation(df, target)
    
    mi_scores['correlation'] = mi_scores['feature'].map(correlations)
    
    print("\nTop 20 features by MI score:")
    print(mi_scores.head(20).to_string(index=False))
    
    # Filter out features with MI ≈ 0
    mi_scores_filtered = mi_scores[mi_scores['mi_score'] >= 0.001].copy()
    print(f"\nRemoved {len(mi_scores) - len(mi_scores_filtered)} features with MI < 0.001")
    
    # Select top K features
    if len(mi_scores_filtered) > top_k:
        selected_mi = mi_scores_filtered.head(top_k)
        print(f"Selecting top {top_k} features by MI score")
    else:
        selected_mi = mi_scores_filtered
        print(f"Keeping all {len(selected_mi)} features (less than top_k={top_k})")
    
    selected_features = selected_mi['feature'].tolist()
    
    # Filter dataframe to selected features + target
    df_selected = df[[target] + selected_features].copy()
    
    # Update categorical and numeric lists
    categorical_cols_selected = [col for col in selected_features if col in categorical_cols]
    numeric_cols_selected = [col for col in selected_features if col in numeric_cols]
    
    print(f"\n✓ Feature selection complete")
    print(f"  - Total features before: {len(feature_cols)}")
    print(f"  - Total features after: {len(selected_features)}")
    print(f"  - Numeric: {len(numeric_cols_selected)}")
    print(f"  - Categorical: {len(categorical_cols_selected)}")
    
    # Track counts for funnel chart
    selection_counts = {
        'before_mi': len(feature_cols),
        'removed_low_mi': len(mi_scores) - len(mi_scores_filtered),
        'after_mi': len(selected_features)
    }
    
    return df_selected, categorical_cols_selected, numeric_cols_selected, selected_mi, selection_counts





def save_outputs(df_categorical, df_encoded, mi_scores, removed_low_info, 
                removed_redundant, output_dir='feature_engineering/output'):
    """
    Save all output files: datasets, reports, and feature importance.
    
    Parameters:
    -----------
    df_categorical : pd.DataFrame
        Dataset with categorical features
    df_encoded : pd.DataFrame
        Dataset with encoded features
    mi_scores : pd.DataFrame
        Feature importance scores
    removed_low_info : list
        Features removed in low-info filtering
    removed_redundant : list
        Features removed in redundancy filtering
    output_dir : str
        Output directory path
    """
    # Save datasets using shared util
    fe_utils.save_datasets(df_categorical, df_encoded, output_dir, '01_fe_statistical')
    
    # Save feature importance
    importance_path = os.path.join(output_dir, '01_fe_statistical_importance.csv')
    mi_scores.to_csv(importance_path, index=False)
    print(f"Saved feature importance: {importance_path}")
    
    # Save selection report
    report_path = os.path.join(output_dir, '01_fe_statistical_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STATISTICAL FEATURE SELECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Final feature count: {df_categorical.shape[1] - 1}\n")
        f.write(f"Final encoded feature count: {df_encoded.shape[1] - 1}\n")
        f.write(f"Target variable: medicaid_only\n")
        f.write(f"Sample size: {len(df_categorical):,}\n\n")
        
        f.write("FEATURES REMOVED - LOW INFORMATION\n")
        f.write("-" * 70 + "\n")
        if removed_low_info:
            for feature, reason in removed_low_info:
                f.write(f"  - {feature}: {reason}\n")
        else:
            f.write("  None\n")
        f.write(f"\nTotal: {len(removed_low_info)}\n\n")
        
        f.write("FEATURES REMOVED - REDUNDANCY (HIGH CORRELATION)\n")
        f.write("-" * 70 + "\n")
        if removed_redundant:
            for feature, reason in removed_redundant:
                f.write(f"  - {feature}: {reason}\n")
        else:
            f.write("  None\n")
        f.write(f"\nTotal: {len(removed_redundant)}\n\n")
        
        f.write("TOP 30 SELECTED FEATURES (BY MI SCORE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<25} {'MI Score':<12} {'Correlation':<12} {'Type':<12}\n")
        f.write("-" * 70 + "\n")
        for idx, row in mi_scores.head(30).iterrows():
            rank = idx + 1
            feature = row['feature']
            mi = row['mi_score']
            corr = row['correlation'] if pd.notna(row['correlation']) else 'N/A'
            ftype = row['feature_type']
            corr_str = f"{corr:.4f}" if corr != 'N/A' else corr
            f.write(f"{rank:<6} {feature:<25} {mi:<12.6f} {corr_str:<12} {ftype:<12}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Mean MI Score: {mi_scores['mi_score'].mean():.6f}\n")
        f.write(f"Median MI Score: {mi_scores['mi_score'].median():.6f}\n")
        f.write(f"Min MI Score: {mi_scores['mi_score'].min():.6f}\n")
        f.write(f"Max MI Score: {mi_scores['mi_score'].max():.6f}\n")
        
    print(f"\n✓ Saved selection report: {report_path}")


def create_visualizations(df_encoded, mi_scores, removed_low_info, removed_redundant, 
                         selection_counts, target='medicaid_only', output_dir='feature_engineering/output'):
    """
    Create and save visualization plots.
    
    Parameters:
    -----------
    df_encoded : pd.DataFrame
        Encoded dataset for correlation heatmap
    mi_scores : pd.DataFrame
        Feature importance scores
    removed_low_info : list
        Features removed in filtering steps
    removed_redundant : list
        Features removed in redundancy step
    selection_counts : dict
        Dictionary with counts at each stage for funnel chart
    target : str
        Target variable name
    output_dir : str
        Output directory path
    """
    print("\n" + "=" * 60)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Feature Importance Bar Chart (Top 30)
    fe_utils.plot_feature_ranking(
        mi_scores['feature'], 
        mi_scores['mi_score'], 
        'Mutual Information Score', 
        os.path.join(output_dir, '01_fe_importance.png')
    )
    
    # 2. Correlation Heatmap (Top 30 features)
    top_30_features = mi_scores.head(30)['feature'].tolist()
    fe_utils.plot_correlation_heatmap(
        df_encoded, 
        os.path.join(output_dir, '01_fe_correlation_heatmap.png'),
        top_features=top_30_features
    )
    
    # 3. Selection Funnel Chart
    print("\nCreating selection funnel chart...")
    
    # Calculate stages using tracked counts
    final_features = selection_counts['after_mi']
    before_mi = selection_counts['before_mi']
    after_redundancy = before_mi
    after_low_info = after_redundancy + len(removed_redundant)
    initial_features = after_low_info + len(removed_low_info)
    
    stages = ['Initial\nFeatures', 'After Low-Info\nFiltering', 
              'After Redundancy\nFiltering', 'After MI\nSelection']
    counts = [initial_features, after_low_info, after_redundancy, final_features]
    
    plt.figure(figsize=(12, 7))
    colors_funnel = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(stages, counts, color=colors_funnel, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Feature Selection Funnel', fontsize=14, fontweight='bold')
    plt.ylim(0, initial_features * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    funnel_path = os.path.join(output_dir, '01_fe_selection_funnel.png')
    plt.savefig(funnel_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {funnel_path}")
    
    print("\n✓ All visualizations created successfully")





def main():
    """Main execution function."""
    
    print("\n" + "=" * 80)
    print("STATISTICAL FEATURE SELECTION FOR PUMS CENSUS DATA")
    print("Predicting Medicaid-Only Insurance Status")
    print("=" * 80)
    
    # Configuration
    input_file = 'data/processed/psam_p38_cleaned.csv'
    output_dir = 'feature_engineering/output'
    top_k_features = 70
    
    # Step 1: Load and prepare data
    df = fe_utils.load_and_prepare_data(input_file)
    
    # Step 2: Identify categorical variables
    df, categorical_cols, numeric_cols = fe_utils.run_preprocessing_pipeline(df)
    
    # Step 3A: Filter low-information features
    df, categorical_cols, numeric_cols, removed_low_info = filter_low_information(
        df, categorical_cols, numeric_cols
    )
    
    # Step 3B: Filter redundant features
    df, numeric_cols, removed_redundant = filter_redundant_features(
        df, categorical_cols, numeric_cols
    )
    
    # Step 3C: Select features by mutual information
    df_selected, categorical_cols_final, numeric_cols_final, mi_scores, selection_counts = \
        select_features_by_mutual_information(
            df, categorical_cols, numeric_cols, top_k=top_k_features
        )
    
    # Create encoded version
    df_encoded = fe_utils.create_encoded_dataset(df_selected, categorical_cols_final)
    
    # Save outputs
    save_outputs(df_selected, df_encoded, mi_scores, removed_low_info, 
                removed_redundant, output_dir)
    
    # Create visualizations
    create_visualizations(df_encoded, mi_scores, removed_low_info, removed_redundant, 
                         selection_counts, output_dir=output_dir)
    
    # Print verification
    fe_utils.print_verification(df_encoded)
    
    print("\n" + "=" * 80)
    print("✓ STATISTICAL FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - 01_fe_statistical_categorical.csv")
    print("  - 01_fe_statistical_encoded.csv")
    print("  - 01_fe_statistical_importance.csv")
    print("  - 01_fe_statistical_report.txt")
    print("  - 01_fe_importance.png")
    print("  - 01_fe_correlation_heatmap.png")
    print("  - 01_fe_selection_funnel.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
