"""
Utility Functions
Helper functions for data processing, evaluation, and visualization
Author: Sneha Vasudev
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class MetricsCalculator:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_task_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_name: str
    ) -> Dict[str, float]:
        """Calculate metrics for single task"""
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score
        )
        
        metrics = {
            'task': task_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    @staticmethod
    def calculate_multi_task_metrics(
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        task_names: List[str],
        task_weights: List[float]
    ) -> Dict:
        """Calculate weighted multi-task metrics"""
        
        from sklearn.metrics import f1_score
        
        task_metrics = []
        f1_scores = []
        
        for i, task_name in enumerate(task_names):
            f1 = f1_score(Y_true[:, i], Y_pred[:, i], average='weighted', zero_division=0)
            task_metrics.append({
                'task': task_name,
                'f1': f1
            })
            f1_scores.append(f1)
        
        weighted_f1 = np.average(f1_scores, weights=task_weights)
        
        return {
            'per_task': task_metrics,
            'weighted_f1': weighted_f1,
            'f1_scores': dict(zip(task_names, f1_scores))
        }
    
    @staticmethod
    def generate_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: List[str] = None
    ) -> str:
        """Generate detailed classification report"""
        
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )


class ResultsExporter:
    """Export results in various formats"""
    
    @staticmethod
    def export_to_json(data: Dict, filepath: Path) -> Path:
        """Export results to JSON"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    @staticmethod
    def export_to_pickle(data: Any, filepath: Path) -> Path:
        """Export results to pickle"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filepath: Path) -> Path:
        """Export results to CSV"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        return filepath
    
    @staticmethod
    def export_predictions(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        email_ids: List[str],
        filepath: Path,
        task_names: List[str] = None
    ) -> Path:
        """Export predictions to CSV"""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'email_id': email_ids,
        }
        
        # Add predictions
        if task_names:
            for i, task_name in enumerate(task_names):
                data[f'{task_name}_pred'] = predictions[:, i]
        else:
            for i in range(predictions.shape[1]):
                data[f'task_{i}_pred'] = predictions[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath


class Visualizer:
    """Create visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        filepath: Path = None
    ) -> Path:
        """Plot confusion matrix"""
        
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            xlabel='Predicted Label',
            ylabel='True Label',
            title=title
        )
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {filepath}")
        
        return filepath
    
    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        labels: List[str],
        filepath: Path = None
    ) -> Path:
        """Plot ROC curves for multi-class"""
        
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(labels):
            # One-vs-rest
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.2f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title='ROC Curves',
            xlim=[0, 1],
            ylim=[0, 1]
        )
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved ROC curves to {filepath}")
        
        return filepath
    
    @staticmethod
    def plot_fitness_history(
        fitness_history: List[float],
        filepath: Path = None
    ) -> Path:
        """Plot NEAT fitness evolution"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(fitness_history, linewidth=2, alpha=0.7)
        ax.fill_between(range(len(fitness_history)), fitness_history, alpha=0.3)
        
        ax.set(
            xlabel='Generation',
            ylabel='Fitness Score',
            title='NEAT Population Fitness Evolution',
            grid=True
        )
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved fitness history to {filepath}")
        
        return filepath


class DataProcessor:
    """Utility functions for data processing"""
    
    @staticmethod
    def create_batches(
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create mini-batches"""
        
        batches = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            batches.append((X_batch, Y_batch))
        
        return batches
    
    @staticmethod
    def normalize_features(
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features using training statistics"""
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        return X_train_norm, X_test_norm


class Logger:
    """Logging utilities"""
    
    @staticmethod
    def log_experiment(
        experiment_name: str,
        results: Dict,
        filepath: Path = None
    ) -> Path:
        """Log experiment results"""
        
        if filepath is None:
            filepath = Path('experiments.log')
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("-"*80 + "\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        return filepath


if __name__ == "__main__":
    print("Utilities module imported successfully")
