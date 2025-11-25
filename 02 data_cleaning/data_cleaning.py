"""
PUMS Data Cleaning Pipeline
============================
This script cleans the PUMS person-level data for machine learning.

Steps:
1. Load Data
Reads the raw CSV file (psam_p38.csv)
Counts total rows and columns

2. Explore Data
Shows the dataset shape (rows x columns)
Lists all column names
Counts data types (how many numeric, categorical, etc.)
Identifies missing values and their percentages
Displays first few rows

3. Handle Missing Values
Drops columns with more than 50% missing data
Fills numeric columns (like age, income, wages) with the median value
Fills categorical columns (like sex, marital status) with the mode (most common value)

4. Convert Data Types
Converts categorical variables (SEX, MAR, RAC1P, HISP, CIT, ESR, SCHL, etc.) to category type
Converts numeric variables (AGEP, WAGP, PINCP, WKHP, etc.) to proper numeric type
This makes the data more memory-efficient and appropriate for analysis

5. Remove Duplicates
Checks for duplicate rows (identical records)
Removes any duplicates found

6. Handle Outliers
Uses the IQR (Interquartile Range) method to detect outliers
By default, focuses on income/wage columns (WAGP, PINCP, PERNP)
Removes rows where values fall outside: Q1 - 3xIQR to Q3 + 3xIQR
This removes extreme values that could skew your ML model

7. Validate Data
Age check: Removes ages outside 0-120 range
Income check: Removes negative wages (WAGP < 0)
Income check: Removes negative total income (PINCP < 0)
Ensures data quality for ML

8. Save Cleaned Data
Saves the cleaned dataset to psam_p38_cleaned.csv
Prints statistics showing:
Original vs cleaned shape
How many rows were removed
How many columns were removed
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PUMSDataCleaner:
    """Clean PUMS person-level data for ML"""
    
    def __init__(self, input_file, output_dir='data/processed'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.df_cleaned = None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load raw data"""
        print("Loading data...")
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
        return self
    
    def explore_data(self):
        """Print basic data exploration"""
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        print(f"\nShape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes.value_counts()}")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing': missing.values,
                'Percentage': missing_pct.values
            })
            print(f"\nMissing values:\n{missing_df[missing_df['Missing'] > 0].to_string(index=False)}")
        else:
            print("\nNo missing values found")
        
        print(f"\nFirst few rows:\n{self.df.head()}")
        return self
    
    def handle_missing_values(self, threshold=0.5):
        """Handle missing values"""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        self.df_cleaned = self.df.copy()
        
        # Remove columns with too many missing values
        missing_pct = self.df_cleaned.isnull().sum() / len(self.df_cleaned)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            print(f"\nDropping {len(cols_to_drop)} columns with >{threshold*100}% missing:")
            print(cols_to_drop)
            self.df_cleaned = self.df_cleaned.drop(columns=cols_to_drop)
        
        # Impute numerical columns with median
        numerical_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} with median: {median_val}")
        
        # Impute categorical columns with mode
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()[0]
                self.df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"Imputed {col} with mode: {mode_val}")
        
        return self
    
    def convert_data_types(self):
        """Convert columns to appropriate data types"""
        print("\n" + "="*80)
        print("CONVERTING DATA TYPES")
        print("="*80)
        
        # Common PUMS categorical variables (convert if they exist)
        categorical_vars = ['SEX', 'MAR', 'RAC1P', 'HISP', 'CIT', 'ESR', 
                           'SCHL', 'SCH', 'COW', 'RELP', 'DIS', 'DEAR', 
                           'DEYE', 'DREM', 'LANX', 'ENG']
        
        for col in categorical_vars:
            if col in self.df_cleaned.columns:
                self.df_cleaned[col] = self.df_cleaned[col].astype('category')
                print(f"Converted {col} to category")
        
        # Ensure numeric columns are numeric
        numeric_vars = ['AGEP', 'WAGP', 'PINCP', 'WKHP', 'POVPIP', 'JWMNP']
        for col in numeric_vars:
            if col in self.df_cleaned.columns:
                self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors='coerce')
                print(f"Converted {col} to numeric")
        
        return self
    
    def handle_outliers(self, columns=None, factor=3.0):
        """Remove outliers using IQR method"""
        print("\n" + "="*80)
        print("HANDLING OUTLIERS")
        print("="*80)
        
        if columns is None:
            # Default to income and wage columns
            columns = [col for col in ['WAGP', 'PINCP', 'PERNP'] 
                      if col in self.df_cleaned.columns]
        
        initial_rows = len(self.df_cleaned)
        
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
                    print(f"{col}: Removed {removed} outliers (range: {lower_bound:.2f} - {upper_bound:.2f})")
        
        total_removed = initial_rows - len(self.df_cleaned)
        print(f"\nTotal rows removed: {total_removed}")
        
        return self
    
    # Feature engineering removed - focusing on data cleaning only
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print("\n" + "="*80)
        print("REMOVING DUPLICATES")
        print("="*80)
        
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows")
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            print(f"Removed duplicates")
        else:
            print("No duplicates found")
        
        return self
    
    def validate_data(self):
        """Validate cleaned data"""
        print("\n" + "="*80)
        print("DATA VALIDATION")
        print("="*80)
        
        issues = []
        
        # Age validation
        if 'AGEP' in self.df_cleaned.columns:
            invalid_age = (~self.df_cleaned['AGEP'].between(0, 120)).sum()
            if invalid_age > 0:
                issues.append(f"Invalid ages: {invalid_age}")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['AGEP'].between(0, 120)]
        
        # Income validation
        if 'WAGP' in self.df_cleaned.columns:
            negative_wage = (self.df_cleaned['WAGP'] < 0).sum()
            if negative_wage > 0:
                issues.append(f"Negative wages: {negative_wage}")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['WAGP'] >= 0]
        
        if 'PINCP' in self.df_cleaned.columns:
            negative_income = (self.df_cleaned['PINCP'] < 0).sum()
            if negative_income > 0:
                issues.append(f"Negative income: {negative_income}")
                self.df_cleaned = self.df_cleaned[self.df_cleaned['PINCP'] >= 0]
        
        if issues:
            print("Issues found and corrected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All validation checks passed!")
        
        return self
    
    def save_cleaned_data(self, filename=None):
        """Save cleaned data"""
        print("\n" + "="*80)
        print("SAVING CLEANED DATA")
        print("="*80)
        
        if filename is None:
            input_name = Path(self.input_file).stem
            filename = f"{input_name}_cleaned.csv"
        
        output_path = Path(self.output_dir) / filename
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        # Print final statistics
        print(f"\nOriginal shape: {self.df.shape}")
        print(f"Cleaned shape: {self.df_cleaned.shape}")
        print(f"Rows removed: {self.df.shape[0] - self.df_cleaned.shape[0]}")
        print(f"Columns removed: {self.df.shape[1] - self.df_cleaned.shape[1]}")
        
        return self
    
    def run_pipeline(self):
        """Run the complete cleaning pipeline"""
        print("="*80)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*80)
        
        (self.load_data()
         .explore_data()
         .handle_missing_values()
         .convert_data_types()
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
        input_file='data/raw/psam_p38.csv',
        output_dir='data/processed'
    )
    
    cleaned_data = cleaner.run_pipeline()
    
    print("\nCleaned data is ready for machine learning!")
