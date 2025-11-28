from sklearn.linear_model import LogisticRegression
from model_utils import ModelTrainer

def main():
    trainer = ModelTrainer(experiment_name="LogisticRegression")
    
    # All datasets
    datasets = [
        "statistical_encoded", "business_encoded", "realistic_encoded",
        "pca_statistical", "pca_business", "pca_realistic"
    ]
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    
    for ds_name in datasets:
        try:
            X, y = trainer.load_data(ds_name)
            trainer.run_training(
                model=LogisticRegression(max_iter=1000),
                param_grid=param_grid,
                dataset_name=ds_name,
                X=X,
                y=y,
                cv_folds=5
            )
        except Exception as e:
            print(f"Failed to run {ds_name}: {e}")

if __name__ == "__main__":
    main()
