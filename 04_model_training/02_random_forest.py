from sklearn.ensemble import RandomForestClassifier
from model_utils import ModelTrainer, DATASETS

def main():
    trainer = ModelTrainer(experiment_name="RandomForest")
    
    # All datasets
    datasets = [
        "statistical_encoded", "business_encoded", "realistic_encoded",
        "pca_statistical", "pca_business", "pca_realistic"
    ]
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    for ds_name in datasets:
        try:
            X, y = trainer.load_data(ds_name)
            trainer.run_training(
                model=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                dataset_name=ds_name,
                X=X,
                y=y,
                cv_folds=5,
                search_type='random'
            )
        except Exception as e:
            print(f"Failed to run {ds_name}: {e}")

if __name__ == "__main__":
    main()
