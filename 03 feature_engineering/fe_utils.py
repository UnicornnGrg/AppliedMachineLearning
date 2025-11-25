"""
Shared utilities for Feature Engineering scripts.
Provides unified data loading, preprocessing, encoding, and visualization functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import logging
from typing import List, Tuple, Dict, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up plotting style and warnings."""
    warnings.filterwarnings('ignore')
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def load_and_prepare_data(filepath: str, target_col: str = 'medicaid_only') -> pd.DataFrame:
    """
    Standard data loading and preparation for PUMS data.
    - Loads CSV
    - Filters Age 18-65
    - Creates Target
    - Drops standard unnecessary columns (weights, flags, insurance details)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: LOAD AND PREPARE DATA (SHARED)")
    logger.info("=" * 60)
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Input file not found: {filepath}")

    logger.info(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise

    logger.info(f"Initial shape: {df.shape}")
    
    # Filter Age
    if 'AGEP' in df.columns:
        logger.info("Filtering to ages 18-65...")
        df = df[(df['AGEP'] >= 18) & (df['AGEP'] <= 65)].copy()
    else:
        logger.warning("Column 'AGEP' not found. Skipping age filtering.")
    
    # Create Target
    logger.info(f"Creating target variable: {target_col}")
    if 'HINS4' in df.columns and 'PRIVCOV' in df.columns:
        df[target_col] = ((df['HINS4'] == 1) & (df['PRIVCOV'] == 2)).astype(int)
    else:
        logger.warning("Columns 'HINS4' or 'PRIVCOV' missing. Cannot create target variable.")
    
    # Stats
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().sort_index()
        logger.info(f"\nTarget distribution:\n{target_counts}")
        logger.info(f"Rate: {df[target_col].mean():.2%}")
    
    # Drop Columns
    cols_to_drop = []
    # Weights
    cols_to_drop.extend([c for c in df.columns if c.startswith('PWGTP')])
    # Flags
    cols_to_drop.extend([c for c in df.columns if c.startswith('F')])
    # IDs and Insurance details
    drop_list = ['RT', 'SERIALNO', 'SPORDER', 'HINS1', 'HINS2', 'HINS3', 
                 'HINS4', 'HINS5', 'HINS6', 'HINS7', 'PUBCOV', 'PRIVCOV', 
                 'HICOV', 'NAICSP', 'SOCP']
    cols_to_drop.extend([c for c in drop_list if c in df.columns])
    
    df = df.drop(columns=list(set(cols_to_drop)), errors='ignore')
    logger.info(f"Dropped {len(cols_to_drop)} columns.")
    logger.info(f"Final shape: {df.shape}")
    
    return df

def process_high_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Standard processing for OCCP and INDP to 2-digit codes."""
    logger.info("\nProcessing High Cardinality Features (OCCP/INDP)...")
    
    for col in ['OCCP', 'INDP']:
        if col in df.columns:
            new_col = f'{col}_2digit'
            # Convert to string, zero pad, take first 2 chars
            df[new_col] = df[col].fillna(0).astype(int).astype(str).str.zfill(4).str[:2]
            logger.info(f"  Converted {col} -> {new_col} ({df[new_col].nunique()} categories)")
            df = df.drop(columns=[col])
            
    return df

# Known categorical variables for PUMS data
PUMS_KNOWN_CATEGORICALS = [
    'SEX', 'MAR', 'ESR', 'COW', 'SCHL', 'SCH', 'DIS', 'DDRS', 'DEAR', 
    'DEYE', 'DOUT', 'DPHY', 'DREM', 'CIT', 'LANX', 'MIL', 'RELSHIPP', 
    'MIG', 'NATIVITY', 'RAC1P', 'HISP', 'WKL', 'WRK', 'REGION', 'DIVISION', 
    'STATE', 'PUMA', 'GCL', 'QTRBIR', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN', 
    'RACBLK', 'RACNH', 'RACNUM', 'RACPI', 'RACSOR', 'RACWHT', 'RC', 
    'ANC', 'ANC1P', 'ANC2P', 'MSP', 'OCCP', 'INDP'
]

def identify_variable_types(df: pd.DataFrame, target_col: str = 'medicaid_only', 
                            unique_threshold: int = 20, known_categoricals: Optional[List[str]] = None, 
                            verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numeric variables based on dtype and cardinality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column to exclude
    unique_threshold : int
        Threshold for numeric columns to be considered categorical if unique values < threshold
    known_categoricals : list
        List of columns known to be categorical. Defaults to PUMS_KNOWN_CATEGORICALS.
    verbose : bool
        If True, prints breakdown of categorical variables by cardinality.
        
    Returns:
    --------
    categorical_cols : list
    numeric_cols : list
    """
    logger.info("\nIdentifying Variable Types (Shared)...")
    
    if known_categoricals is None:
        known_categoricals = PUMS_KNOWN_CATEGORICALS
        
    # Start with object/category dtypes
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Add known categoricals if present
    for col in known_categoricals:
        if col in df.columns and col not in categorical_cols:
            categorical_cols.append(col)
            
    # Check numeric columns for low cardinality
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_candidates:
        numeric_candidates.remove(target_col)
        
    for col in numeric_candidates:
        if col in categorical_cols:
            continue
            
        n_unique = df[col].nunique()
        if n_unique < unique_threshold:
            logger.info(f"  Reclassifying {col} as categorical ({n_unique} unique values)")
            categorical_cols.append(col)
            
    # Finalize lists
    categorical_cols = list(set(categorical_cols))
    # Ensure target is not included
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != target_col]
    
    logger.info(f"  Identified {len(categorical_cols)} categorical and {len(numeric_cols)} numeric features.")
    
    if verbose:
        logger.info("\nCategorical Variable Breakdown:")
        breakdown = {'Binary': 0, 'Low (3-15)': 0, 'Medium (16-50)': 0, 'High (>50)': 0}
        for col in categorical_cols:
            n = df[col].nunique()
            if n == 2: breakdown['Binary'] += 1
            elif n <= 15: breakdown['Low (3-15)'] += 1
            elif n <= 50: breakdown['Medium (16-50)'] += 1
            else: breakdown['High (>50)'] += 1
        for k, v in breakdown.items():
            logger.info(f"  {k}: {v}")

    return sorted(categorical_cols), sorted(numeric_cols)

def run_preprocessing_pipeline(df: pd.DataFrame, target_col: str = 'medicaid_only') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Run the standard preprocessing pipeline:
    1. Process high cardinality features (OCCP/INDP)
    2. Identify variable types
    3. Handle missing values
    
    Returns:
    --------
    df : pd.DataFrame
        Processed dataframe
    categorical_cols : list
    numeric_cols : list
    """
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING STANDARD PREPROCESSING PIPELINE (SHARED)")
    logger.info("=" * 60)
    
    # 1. High Cardinality
    df = process_high_cardinality(df)
    
    # 2. Identify Types
    categorical_cols, numeric_cols = identify_variable_types(df, target_col=target_col, verbose=True)
    
    # 3. Handle Missing
    df = handle_missing_values(df, numeric_cols, categorical_cols)
    
    return df, categorical_cols, numeric_cols

def handle_missing_values(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, 
                          categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standard missing value handling:
    - Numeric: Fill with median
    - Categorical: Fill with 'Missing'
    """
    logger.info("\nHandling Missing Values (Shared)...")
    
    if numeric_cols is None or categorical_cols is None:
        # Auto-detect if not provided
        cat_cols, num_cols = identify_variable_types(df)
        if numeric_cols is None: numeric_cols = num_cols
        if categorical_cols is None: categorical_cols = cat_cols
        
    # Numeric
    for col in numeric_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            median_val = df[col].median()
            count = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: Filled {count} missing with median ({median_val:.2f})")
            
    # Categorical
    for col in categorical_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            count = df[col].isna().sum()
            df[col] = df[col].fillna('Missing')
            logger.info(f"  {col}: Filled {count} missing with 'Missing'")
            
    return df

def calculate_target_correlation(df: pd.DataFrame, target_col: str = 'medicaid_only') -> Dict[str, float]:
    """Calculate correlation of all numeric features with target."""
    logger.info("\nCalculating Target Correlations (Shared)...")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return {}
        
    correlations = numeric_df.corrwith(numeric_df[target_col]).sort_values(ascending=False)
    # Remove target itself
    correlations = correlations.drop(target_col, errors='ignore')
    
    logger.info("Top 10 Positive Correlations:")
    logger.info(correlations.head(10))
    logger.info("\nTop 10 Negative Correlations:")
    logger.info(correlations.tail(10))
    
    return correlations.to_dict()

def create_encoded_dataset(df: pd.DataFrame, categorical_cols: List[str], target_col: str = 'medicaid_only') -> pd.DataFrame:
    """Standard one-hot encoding using pandas get_dummies."""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING ENCODED DATASET (SHARED)")
    logger.info("=" * 60)
    
    # Separate target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        logger.warning(f"Target column '{target_col}' not found. Encoding all columns.")
        y = None
        X = df
    
    # Encode
    logger.info(f"Encoding {len(categorical_cols)} categorical features...")
    # Ensure dtype=int for 0/1 output
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Recombine
    df_encoded = X_encoded.copy()
    if y is not None:
        df_encoded[target_col] = y
    
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Encoded features: {X_encoded.shape[1]}")
    logger.info(f"Total columns: {df_encoded.shape[1]}")
    
    return df_encoded

def save_datasets(df_categorical: pd.DataFrame, df_encoded: pd.DataFrame, output_dir: str, prefix: str):
    """Save standard categorical and encoded datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("SAVING OUTPUTS (SHARED)")
    logger.info("=" * 60)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        cat_path = os.path.join(output_dir, f'{prefix}_categorical.csv')
        df_categorical.to_csv(cat_path, index=False)
        logger.info(f"Saved categorical: {cat_path}")
        
        enc_path = os.path.join(output_dir, f'{prefix}_encoded.csv')
        df_encoded.to_csv(enc_path, index=False)
        logger.info(f"Saved encoded: {enc_path}")
    except Exception as e:
        logger.error(f"Failed to save datasets: {e}")
        raise

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str, top_features: Optional[List[str]] = None, 
                             target_col: str = 'medicaid_only'):
    """
    Standard correlation heatmap.
    If top_features is provided (list of names), filters to those.
    Otherwise plots all numeric.
    """
    logger.info(f"\nGenerating Correlation Heatmap...")
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if top_features:
        # Filter to top features that exist in numeric_df
        cols_to_plot = [c for c in top_features if c in numeric_df.columns]
        # Always include target if present
        if target_col in numeric_df.columns and target_col not in cols_to_plot:
            cols_to_plot.append(target_col)
        numeric_df = numeric_df[cols_to_plot]
    
    if numeric_df.shape[1] > 50:
        logger.warning("Too many features for heatmap. Selecting top 50 by correlation with target.")
        if target_col in numeric_df.columns:
            corrs = numeric_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
            top_50 = corrs.head(50).index.tolist()
            numeric_df = numeric_df[top_50]
    
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=False)
    
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")

def plot_feature_ranking(features: Union[List[str], pd.Series], scores: Union[List[float], pd.Series], 
                         metric_name: str, output_path: str, color: str = 'steelblue', top_n: int = 30):
    """Standard horizontal bar chart for feature ranking."""
    logger.info(f"\nGenerating Feature Ranking Plot...")
    
    data = pd.DataFrame({'feature': features, 'score': scores})
    # Sort by score
    data = data.sort_values('score', ascending=True)
    
    if len(data) > top_n:
        data = data.tail(top_n)
        
    plt.figure(figsize=(10, 12))
    plt.barh(data['feature'], data['score'], color=color)
    plt.xlabel(metric_name)
    plt.title(f'Top {len(data)} Features by {metric_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")

def print_verification(df_encoded: pd.DataFrame, target_col: str = 'medicaid_only'):
    """Standard verification of the final dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATASET VERIFICATION (SHARED)")
    logger.info("=" * 80)
    
    logger.info(f"Final shape: {df_encoded.shape}")
    logger.info(f"Target: {target_col}")
    
    # Check numeric
    non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"WARNING: Non-numeric columns found: {non_numeric}")
        logger.info("Ready for modeling: NO")
    else:
        logger.info("All columns are numeric.")
        logger.info("Ready for modeling: YES")
    logger.info("=" * 80)

def apply_pca(df: pd.DataFrame, target_col: str = 'medicaid_only', n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA, StandardScaler, List[str]]:
    """
    Apply PCA to the dataset.
    1. Standardize features
    2. Apply PCA
    3. Return transformed dataframe, PCA object, Scaler, and feature names
    """
    logger.info("\n" + "=" * 60)
    logger.info("APPLYING PCA (SHARED)")
    logger.info("=" * 60)
    
    # Separate target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
        
    feature_names = X.columns.tolist()
    
    # Standardize
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    logger.info(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    if y is not None:
        df_pca[target_col] = y.reset_index(drop=True)
        
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"PCA components: {X_pca.shape[1]}")
    logger.info(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return df_pca, pca, scaler, feature_names

def save_pca_models(pca: PCA, scaler: StandardScaler, output_dir: str, prefix: str):
    """Save PCA model and scaler."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(pca, os.path.join(output_dir, f'{prefix}_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, f'{prefix}_scaler.pkl'))
        logger.info(f"Saved PCA model and scaler to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save PCA models: {e}")

def plot_pca_scree(pca: PCA, output_path: str):
    """Plot Scree Plot."""
    logger.info(f"\nGenerating Scree Plot...")
    
    n_components = min(30, len(pca.explained_variance_ratio_))
    variance = pca.explained_variance_ratio_[:n_components]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_components + 1), variance, 'bo-', linewidth=2)
    plt.title('Scree Plot (First 30 Components)')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")

def plot_pca_cumulative_variance(pca: PCA, output_path: str):
    """Plot Cumulative Variance."""
    logger.info(f"\nGenerating Cumulative Variance Plot...")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', linewidth=2)
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")

def analyze_pca_loadings(pca: PCA, feature_names: List[str], n_components: int = 10) -> pd.DataFrame:
    """
    Analyze PCA loadings to interpret components.
    Returns DataFrame of loadings.
    """
    logger.info("\nAnalyzing PCA Loadings...")
    
    n_comps = min(n_components, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_comps].T,
        columns=[f'PC{i+1}' for i in range(n_comps)],
        index=feature_names
    )
    
    return loadings

def plot_pca_loadings_heatmap(loadings_df: pd.DataFrame, output_path: str, n_features: int = 30):
    """Plot heatmap of top feature loadings."""
    logger.info(f"\nGenerating Loadings Heatmap...")
    
    # Select top features by max absolute loading across displayed components
    loadings_abs = loadings_df.abs().max(axis=1)
    top_features = loadings_abs.sort_values(ascending=False).head(n_features).index
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(loadings_df.loc[top_features], cmap='coolwarm', center=0, annot=False)
    plt.title(f'PCA Loadings Heatmap (Top {n_features} Features)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")
