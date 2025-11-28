from sklearn.neural_network import MLPClassifier
from model_utils import ModelTrainer, DATASETS

def main():
    trainer = ModelTrainer(experiment_name="NeuralNetwork")
    
    # All datasets
    datasets = [
        "statistical_encoded", "business_encoded", "realistic_encoded",
        "pca_statistical", "pca_business", "pca_realistic"
    ]
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
    
    for ds_name in datasets:
        try:
            X, y = trainer.load_data(ds_name)
            trainer.run_training(
                model=MLPClassifier(max_iter=500, random_state=42),
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
