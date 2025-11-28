from lightgbm import LGBMClassifier
from model_utils import ModelTrainer, DATASETS

def main():
    trainer = ModelTrainer(experiment_name="LightGBM")
    
    datasets = [
        "statistical_encoded", "business_encoded", "realistic_encoded",
        "pca_statistical", "pca_business", "pca_realistic"
    ]
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50],
        'max_depth': [-1, 10, 20]
    }
    
    for ds_name in datasets:
        try:
            X, y = trainer.load_data(ds_name)
            trainer.run_training(
                model=LGBMClassifier(random_state=42),
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
