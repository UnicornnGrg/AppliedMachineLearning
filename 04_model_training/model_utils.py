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
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Evaluator
        self.evaluator = ModelEvaluator(self.output_dir)
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Hyperparameter Tuning
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1, verbose=1, n_iter=20, random_state=42)
            
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        
        # Evaluation (Delegated to ModelEvaluator)
        metrics = self.evaluator.evaluate(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name,
            experiment_name=self.experiment_name,
            best_params=search.best_params_
        )
        
        # Save Model Artifact
        self._save_model(best_model, dataset_name)
        return metrics

    def _save_model(self, model, dataset_name):
        # Save Model
        model_path = os.path.join(self.output_dir, f"{self.experiment_name}_{dataset_name}_model.pkl")
        joblib.dump(model, model_path)
