import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import joblib
from evaluation import ModelEvaluator

# Dataset Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03 feature_engineering", "output")

DATASETS = {
    "statistical_encoded": "01_fe_statistical_encoded.csv",
    "business_encoded": "02_fe_business_encoded.csv",
    "realistic_encoded": "03_fe_realistic_encoded.csv",
    "pca_statistical": "04_fe_pca_statistical.csv",
    "pca_business": "05_fe_pca_business.csv",
    "pca_realistic": "06_fe_pca_realistic.csv"
}

class ModelTrainer:
    def __init__(self, experiment_name, output_dir=None):
        self.experiment_name = experiment_name
        if output_dir is None:
            base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        else:
            base_output_dir = output_dir
            
        # Create subdirectories
        self.output_dir = base_output_dir
        self.models_dir = os.path.join(base_output_dir, "models")
        self.reports_dir = os.path.join(base_output_dir, "reports")
        self.plots_dir = os.path.join(base_output_dir, "plots")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize Evaluator with specific paths
        self.evaluator = ModelEvaluator(base_output_dir, self.reports_dir, self.plots_dir)
        
    def load_data(self, dataset_key, target_col='medicaid_only'):
        filename = DATASETS.get(dataset_key)
        if not filename:
            raise ValueError(f"Dataset key {dataset_key} not found in configuration.")
            
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Loading data from {filepath}...")
        
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataset.")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    def run_training(self, model, param_grid, dataset_name, X, y, cv_folds=5, search_type='grid'):
        print(f"Starting training for {self.experiment_name} on {dataset_name}...")
        
        # Split data: 70% Train, 15% Val, 15% Test
        # 1. Split off Test (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        
        # 2. Split remaining (85%) into Train (70% total) and Val (15% total)
        # Val size relative to temp = 0.15 / 0.85 = ~0.1765
        val_size = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)
        
        print(f"Data Split Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # Hyperparameter Tuning (on 70% Train)
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1, verbose=1, n_iter=20, random_state=42)
            
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        
        # Evaluation on Validation Set
        metrics_val = self.evaluator.evaluate(
            model=best_model,
            X=X_val,
            y=y_val,
            dataset_name=dataset_name,
            experiment_name=self.experiment_name,
            phase="validation",
            best_params=search.best_params_
        )
        
        print(f"Test set (n={len(X_test)}) reserved for later evaluation.")
        
        # Save Model Artifact
        self._save_model(best_model, dataset_name)
        return metrics_val

    def _save_model(self, model, dataset_name):
        # Save Model
        model_path = os.path.join(self.models_dir, f"{self.experiment_name}_{dataset_name}_model.pkl")
        joblib.dump(model, model_path)
