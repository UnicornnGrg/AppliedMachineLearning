"""
PUMS DATA CLEANING & LEAKAGE HANDLING PIPELINE
=============================================
Goal:
  - Clean PUMS person-level data for machine learning while explicitly handling data leakage.
  - Ensure the output is suitable for predictive modeling (e.g., health insurance coverage)
    by removing features that would leak information about the target variable.
  - Provide flexibility to retain leakage columns for exploratory analysis if needed.
  - Generate a detailed cleaning report in cleaning_report.txt

What this script does:
  1) Load raw PUMS data (person records).
  2) Explore data structure, missing values, and basic statistics.
  3) Handle missing values:
      * Drop columns with >50% missing data
      * Impute numeric columns with median
      * Impute categorical columns with mode
  4) Convert data types:
      * Categorical variables (SEX, MAR, RAC1P, etc.) to category type
      * Numeric variables (AGEP, WAGP, PINCP, etc.) to proper numeric type
  5) Handle data leakage:
      * Identify and remove columns that directly define the target (HICOV)
      * Leakage columns: PRIVCOV, PUBCOV, HINS1-HINS7 (direct components of insurance coverage)
      * Option to keep leakage columns for exploratory analysis
  6) Remove outliers:
      * Uses IQR method on income/wage columns (WAGP, PINCP, PERNP)
      * Removes extreme values that could skew analysis
  7) Remove duplicates:
      * Identifies and removes identical rows
  8) Validate data:
      * Checks for invalid ages (0-120 range)
      * Removes negative wage/income values
  9) Save cleaned data:
      * Outputs to data/processed/psam_p38_cleaned.csv
      * Generates cleaning_report.txt with detailed information about all transformations
      * Reports statistics on rows/columns removed

IMPORTANT:
  - This script focuses on DATA CLEANING and LEAKAGE PREVENTION.
  - It does NOT perform feature engineering (see feature_engineering.py for that).
  - Leakage columns are DROPPED BY DEFAULT to prevent data leakage in models.
  - Use keep_leakage=True in run_pipeline() to retain them for exploratory analysis.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class PUMSDataCleaner:
    """Clean PUMS person-level data for ML, including leakage handling."""

    def __init__(self, input_file, output_dir='data/processed'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.df_cleaned = None
        self.leakage_cols = []  # Track leakage columns
        self.cleaning_report = []  # Store cleaning steps for report
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize report with timestamp and input file info
        self.cleaning_report.append(f"{'='*80}")
        self.cleaning_report.append("PUMS DATA CLEANING REPORT")
        self.cleaning_report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.cleaning_report.append(f"Input file: {input_file}")
        self.cleaning_report.append(f"{'='*80}\n")

    def load_data(self):
        """Load raw data"""
        print("Loading data...")
        self.df = pd.read_csv(self.input_file)
        self.cleaning_report.append(f"Initial data shape: {self.df.shape}")
        self.cleaning_report.append(f"Initial columns: {list(self.df.columns)}\n")
        print(f"Loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
        return self

    def explore_data(self):
        """Print basic data exploration"""
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)

        self.cleaning_report.append(f"{'='*40}")
        self.cleaning_report.append("DATA EXPLORATION")
        self.cleaning_report.append(f"{'='*40}")
        self.cleaning_report.append(f"Shape: {self.df.shape}")
        self.cleaning_report.append(f"Columns: {list(self.df.columns)}")
        self.cleaning_report.append(f"Data types:\n{self.df.dtypes.value_counts().to_string()}")

        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing': missing.values,
                'Percentage': missing_pct.values
            })
            self.cleaning_report.append("\nMissing values:")
            self.cleaning_report.append(missing_df[missing_df['Missing'] > 0].to_string(index=False))
        else:
            self.cleaning_report.append("\nNo missing values found")

        print(f"\nShape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes.value_counts()}")
        if missing.sum() > 0:
            print(f"\nMissing values:\n{missing_df[missing_df['Missing'] > 0].to_string(index=False)}")
        else:
            print("\nNo missing values found")

        return self

    def handle_missing_values(self, threshold=0.5):
        """Handle missing values"""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("HANDLING MISSING VALUES")
        self.cleaning_report.append(f"{'='*40}")

        self.df_cleaned = self.df.copy()
        # Remove columns with too many missing values
        missing_pct = self.df_cleaned.isnull().sum() / len(self.df_cleaned)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

        if cols_to_drop:
            self.cleaning_report.append(f"\nDropped {len(cols_to_drop)} columns with >{threshold*100}% missing:")
            for col in cols_to_drop:
                self.cleaning_report.append(f"  - {col}")
            print(f"\nDropping {len(cols_to_drop)} columns with >{threshold*100}% missing:")
            print(cols_to_drop)
            self.df_cleaned = self.df_cleaned.drop(columns=cols_to_drop)

        # Impute numerical columns with median
        numerical_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        imputed_numeric = []
        for col in numerical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                imputed_numeric.append(f"  - {col}: filled with median={median_val:.2f}")
                print(f"Imputed {col} with median: {median_val}")

        if imputed_numeric:
            self.cleaning_report.append("\nNumeric columns imputed with median:")
            self.cleaning_report.extend(imputed_numeric)

        # Impute categorical columns with mode
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        imputed_categorical = []
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()[0]
                self.df_cleaned[col].fillna(mode_val, inplace=True)
                imputed_categorical.append(f"  - {col}: filled with mode={mode_val}")
                print(f"Imputed {col} with mode: {mode_val}")

        if imputed_categorical:
            self.cleaning_report.append("\nCategorical columns imputed with mode:")
            self.cleaning_report.extend(imputed_categorical)

        self.cleaning_report.append(f"\nShape after handling missing values: {self.df_cleaned.shape}")
        return self

    def convert_data_types(self):
        """Convert columns to appropriate data types"""
        print("\n" + "="*80)
        print("CONVERTING DATA TYPES")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("CONVERTING DATA TYPES")
        self.cleaning_report.append(f"{'='*40}")

        # Common PUMS categorical variables (convert if they exist)
        categorical_vars = ['SEX', 'MAR', 'RAC1P', 'HISP', 'CIT', 'ESR',
                           'SCHL', 'SCH', 'COW', 'RELP', 'DIS', 'DEAR',
                           'DEYE', 'DREM', 'LANX', 'ENG']
        converted_categorical = []
        for col in categorical_vars:
            if col in self.df_cleaned.columns:
                self.df_cleaned[col] = self.df_cleaned[col].astype('category')
                converted_categorical.append(f"  - {col}")
                print(f"Converted {col} to category")

        if converted_categorical:
            self.cleaning_report.append("\nColumns converted to category:")
            self.cleaning_report.extend(converted_categorical)

        # Ensure numeric columns are numeric
        numeric_vars = ['AGEP', 'WAGP', 'PINCP', 'WKHP', 'POVPIP', 'JWMNP']
        converted_numeric = []
        for col in numeric_vars:
            if col in self.df_cleaned.columns:
                self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors='coerce')
                converted_numeric.append(f"  - {col}")
                print(f"Converted {col} to numeric")

        if converted_numeric:
            self.cleaning_report.append("\nColumns converted to numeric:")
            self.cleaning_report.extend(converted_numeric)

        self.cleaning_report.append(f"\nShape after type conversion: {self.df_cleaned.shape}")
        return self

    def handle_leakage(self, keep_leakage=False):
        """
        Handle data leakage by removing columns that directly define the target (HICOV).
        Leakage columns: PRIVCOV, PUBCOV, HINS1-HINS7.
        These columns contain information that would leak the target variable (insurance coverage)
        and must be excluded from predictive modeling to avoid inflated performance metrics.
        """
        print("\n" + "="*80)
        print("HANDLING DATA LEAKAGE")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("HANDLING DATA LEAKAGE")
        self.cleaning_report.append(f"{'='*40}")

        # Define leakage columns (directly related to HICOV)
        leakage_cols = ['PRIVCOV', 'PUBCOV', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7']
        self.leakage_cols = [col for col in leakage_cols if col in self.df_cleaned.columns]

        if not keep_leakage and self.leakage_cols:
            self.cleaning_report.append(f"\nDropped leakage columns (would leak target information):")
            for col in self.leakage_cols:
                self.cleaning_report.append(f"  - {col}")
            print(f"Dropping leakage columns: {self.leakage_cols}")
            print("These columns contain direct information about insurance coverage and would leak target information.")
            self.df_cleaned = self.df_cleaned.drop(columns=self.leakage_cols)
        else:
            self.cleaning_report.append(f"\nKept leakage columns for analysis (WARNING: not safe for modeling):")
            for col in self.leakage_cols:
                self.cleaning_report.append(f"  - {col}")
            print(f"Keeping leakage columns for analysis: {self.leakage_cols}")
            print("WARNING: These columns should NOT be used for predictive modeling as they leak target information.")

        self.cleaning_report.append(f"\nShape after leakage handling: {self.df_cleaned.shape}")
        return self

    def handle_outliers(self, columns=None, factor=3.0):
        """Remove outliers using IQR method"""
        print("\n" + "="*80)
        print("HANDLING OUTLIERS")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("HANDLING OUTLIERS")
        self.cleaning_report.append(f"{'='*40}")

        if columns is None:
            # Default to income and wage columns
            columns = [col for col in ['WAGP', 'PINCP', 'PERNP']
                      if col in self.df_cleaned.columns]

        self.cleaning_report.append(f"\nOutlier detection applied to columns: {columns}")
        self.cleaning_report.append(f"Using IQR method with factor: {factor}")

        initial_rows = len(self.df_cleaned)
        outlier_stats = []

        for col in columns:
            if col in self.df_cleaned.columns:
                Q1 = self.df_cleaned[col].quantile(0.25)
                Q3 = self.df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                before = len(self.df_cleaned)
                self.df_cleaned = self.df_cleaned[
                    (self.df_cleaned[col] >= lower_bound) &
                    (self.df_cleaned[col] <= upper_bound)
                ]
                removed = before - len(self.df_cleaned)
                if removed > 0:
                    outlier_stats.append(f"  - {col}: Removed {removed} outliers (range: {lower_bound:.2f} - {upper_bound:.2f})")
                    print(f"{col}: Removed {removed} outliers (range: {lower_bound:.2f} - {upper_bound:.2f})")

        if outlier_stats:
            self.cleaning_report.append("\nOutliers removed:")
            self.cleaning_report.extend(outlier_stats)

        total_removed = initial_rows - len(self.df_cleaned)
        if total_removed > 0:
            self.cleaning_report.append(f"\nTotal rows removed due to outliers: {total_removed}")
        else:
            self.cleaning_report.append("\nNo outliers removed")

        self.cleaning_report.append(f"\nShape after outlier handling: {self.df_cleaned.shape}")
        return self

    def remove_duplicates(self):
        """Remove duplicate rows"""
        print("\n" + "="*80)
        print("REMOVING DUPLICATES")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("REMOVING DUPLICATES")
        self.cleaning_report.append(f"{'='*40}")

        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            self.cleaning_report.append(f"\nFound and removed {duplicates} duplicate rows")
            print(f"Found {duplicates} duplicate rows")
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            print(f"Removed duplicates")
        else:
            self.cleaning_report.append("\nNo duplicates found")

        self.cleaning_report.append(f"\nShape after duplicate removal: {self.df_cleaned.shape}")
        return self

    def validate_data(self):
        """Validate cleaned data"""
        print("\n" + "="*80)
        print("DATA VALIDATION")
        print("="*80)

        self.cleaning_report.append(f"\n{'='*40}")
        self.cleaning_report.append("DATA VALIDATION")
        self.cleaning_report.append(f"{'='*40}")

        issues = []
        validation_actions = []

        # Age validation
        if 'AGEP' in self.df_cleaned.columns:
            invalid_age = (~self.df_cleaned['AGEP'].between(0, 120)).sum()
            if invalid_age > 0:
                issues.append(f"Invalid ages: {invalid_age}")
                validation_actions.append(f"  - Removed {invalid_age} rows with ages outside 0-120 range")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['AGEP'].between(0, 120)]

        # Income validation
        if 'WAGP' in self.df_cleaned.columns:
            negative_wage = (self.df_cleaned['WAGP'] < 0).sum()
            if negative_wage > 0:
                issues.append(f"Negative wages: {negative_wage}")
                validation_actions.append(f"  - Removed {negative_wage} rows with negative wages")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['WAGP'] >= 0]

        if 'PINCP' in self.df_cleaned.columns:
            negative_income = (self.df_cleaned['PINCP'] < 0).sum()
            if negative_income > 0:
                issues.append(f"Negative income: {negative_income}")
                validation_actions.append(f"  - Removed {negative_income} rows with negative income")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['PINCP'] >= 0]

        if issues:
            self.cleaning_report.append("\nData quality issues found and corrected:")
            self.cleaning_report.extend(issues)
            self.cleaning_report.append("\nActions taken:")
            self.cleaning_report.extend(validation_actions)
            print("Issues found and corrected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            self.cleaning_report.append("\nAll validation checks passed!")
            print("All validation checks passed!")

        self.cleaning_report.append(f"\nShape after validation: {self.df_cleaned.shape}")
        return self

    def save_cleaned_data(self, filename=None):
        """Save cleaned data and cleaning report"""
        print("\n" + "="*80)
        print("SAVING CLEANED DATA")
        print("="*80)

        if filename is None:
            input_name = Path(self.input_file).stem
            filename = f"{input_name}_cleaned.csv"

        output_path = Path(self.output_dir) / filename
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")

        # Save cleaning report
        report_path = Path(self.output_dir) / f"{Path(input_name).stem}_cleaning_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.cleaning_report))

        print(f"Saved cleaning report to: {report_path}")

        # Print final statistics
        self.cleaning_report.append(f"\n{'='*80}")
        self.cleaning_report.append("FINAL SUMMARY")
        self.cleaning_report.append(f"{'='*80}")
        self.cleaning_report.append(f"Original shape: {self.df.shape}")
        self.cleaning_report.append(f"Cleaned shape: {self.df_cleaned.shape}")
        self.cleaning_report.append(f"Total rows removed: {self.df.shape[0] - self.df_cleaned.shape[0]}")
        self.cleaning_report.append(f"Total columns removed: {self.df.shape[1] - self.df_cleaned.shape[1]}")

        if self.leakage_cols and 'PRIVCOV' not in self.df_cleaned.columns:
            self.cleaning_report.append("\nNote: Leakage columns were removed. Data is safe for predictive modeling.")
            print(f"Note: Leakage columns were removed. Data is safe for predictive modeling.")
        else:
            self.cleaning_report.append("\nWARNING: Leakage columns were retained. Data is NOT safe for predictive modeling.")
            print(f"WARNING: Leakage columns were retained. Data is NOT safe for predictive modeling.")

        print(f"\nOriginal shape: {self.df.shape}")
        print(f"Cleaned shape: {self.df_cleaned.shape}")
        print(f"Rows removed: {self.df.shape[0] - self.df_cleaned.shape[0]}")
        print(f"Columns removed: {self.df.shape[1] - self.df_cleaned.shape[1]}")

        return self

    def run_pipeline(self, keep_leakage=False):
        """Run the complete cleaning pipeline"""
        print("="*80)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*80)

        (self.load_data()
         .explore_data()
         .handle_missing_values()
         .convert_data_types()
         .handle_leakage(keep_leakage=keep_leakage)  # Leakage handling
         .remove_duplicates()
         .handle_outliers()
         .validate_data()
         .save_cleaned_data())

        print("\n" + "="*80)
        print("CLEANING COMPLETE!")
        print("="*80)
        return self.df_cleaned

if __name__ == "__main__":
    # Run the cleaning pipeline
    cleaner = PUMSDataCleaner(
        input_file='00 data/raw/psam_p38.csv',
        output_dir='00 data/processed'
    )
    # Set keep_leakage=True to retain leakage columns for exploratory analysis
    # WARNING: Only use keep_leakage=True if you're NOT building predictive models
    cleaned_data = cleaner.run_pipeline(keep_leakage=False)
    print("\nCleaned data is ready for machine learning!")
    print("Remember: Leakage columns were removed by default for safe modeling.")
    print("See cleaning_report.txt for detailed information about all transformations.")
