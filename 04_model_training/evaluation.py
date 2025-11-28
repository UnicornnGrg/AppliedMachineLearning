import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

class ModelEvaluator:
    def __init__(self, output_dir, reports_dir, plots_dir):
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.plots_dir = plots_dir

    def evaluate(self, model, X, y, dataset_name, experiment_name, phase="test", best_params=None):
        """
        Evaluates the model on the provided set (test or validation) and saves metrics.
        """
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate Metrics
        metrics = {
            "dataset": dataset_name,
            "model": experiment_name,
            "phase": phase,
            "best_params": str(best_params) if best_params else "N/A",
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob) if y_prob is not None else None
        }

        # Print Summary
        print(f"Results for {experiment_name} on {dataset_name} ({phase}):")
        print(f"  F1: {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  AUC: {metrics['roc_auc']:.4f}")

        # Save Metrics to CSV
        self._save_metrics_csv(metrics)

        # Save Detailed Report
        self._save_report(experiment_name, dataset_name, phase, metrics, y, y_pred)

        # Save Confusion Matrix
        self._save_confusion_matrix(y, y_pred, experiment_name, dataset_name, phase)

        return metrics

    def _save_metrics_csv(self, metrics):
        metrics_file = os.path.join(self.output_dir, "all_experiment_results.csv")
        metrics_df = pd.DataFrame([metrics])
        
        # Append if exists, else write new
        if os.path.exists(metrics_file):
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_file, mode='w', header=True, index=False)

    def _save_report(self, experiment_name, dataset_name, phase, metrics, y_true, y_pred):
        report_path = os.path.join(self.reports_dir, f"{experiment_name}_{dataset_name}_{phase}_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Best Params: {metrics['best_params']}\n")
            f.write("-" * 30 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n" + "-" * 30 + "\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, zero_division=0))

    def _save_confusion_matrix(self, y_true, y_pred, experiment_name, dataset_name, phase):
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {experiment_name} - {dataset_name} ({phase})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_dir, f"{experiment_name}_{dataset_name}_{phase}_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plot. Error: {e}")
