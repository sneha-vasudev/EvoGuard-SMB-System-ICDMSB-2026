"""
Baseline Models Module
Traditional ML models for performance comparison
Author: Sneha Vasudev
"""

import time
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from scipy import sparse
from typing import Dict, Tuple, Any

warnings.filterwarnings('ignore')


class BaselineModels:
    """Train and evaluate baseline models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}
        self.scaler = None
        
    def train_logistic_regression(
        self, 
        X_train: sparse.csr_matrix, 
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Logistic Regression"""
        print("\n[Baseline] Training Logistic Regression...")
        
        t0 = time.time()
        model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['logistic_regression'] = model
        self.training_times['logistic_regression'] = train_time
        
        result = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.results['logistic_regression'] = result
        print(f"  ✓ F1 Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        return result
    
    def train_random_forest(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Random Forest"""
        print("\n[Baseline] Training Random Forest...")
        
        # Convert sparse to dense for RF (memory permitting)
        X_train_dense = X_train.toarray() if X_train.shape[0] < 100000 else X_train
        X_test_dense = X_test.toarray() if X_test.shape[0] < 100000 else X_test
        
        t0 = time.time()
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_dense, y_train)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test_dense)
        y_proba = model.predict_proba(X_test_dense)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['random_forest'] = model
        self.training_times['random_forest'] = train_time
        
        result = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.results['random_forest'] = result
        print(f"  ✓ F1 Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        return result
    
    def train_mlp(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Multi-Layer Perceptron"""
        print("\n[Baseline] Training MLP...")
        
        # Scale features
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        t0 = time.time()
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
            n_jobs=1
        )
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['mlp'] = model
        self.training_times['mlp'] = train_time
        
        result = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.results['mlp'] = result
        print(f"  ✓ F1 Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        return result
    
    def train_xgboost(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train XGBoost"""
        print("\n[Baseline] Training XGBoost...")
        
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("  ✗ XGBoost not installed. Install with: pip install xgboost")
            return None
        
        # XGBoost works with both sparse and dense
        X_train_dense = X_train.toarray() if sparse.issparse(X_train) else X_train
        X_test_dense = X_test.toarray() if sparse.issparse(X_test) else X_test
        
        t0 = time.time()
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train_dense, y_train, verbose=False)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test_dense)
        y_proba = model.predict_proba(X_test_dense)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['xgboost'] = model
        self.training_times['xgboost'] = train_time
        
        result = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.results['xgboost'] = result
        print(f"  ✓ F1 Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        return result
    
    def train_svm(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Linear SVM with calibration"""
        print("\n[Baseline] Training Linear SVM...")
        
        t0 = time.time()
        svm = LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
            dual='auto',
            verbose=0
        )
        
        # Calibrate for probability estimates
        model = CalibratedClassifierCV(svm, cv=3)
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['svm'] = model
        self.training_times['svm'] = train_time
        
        result = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.results['svm'] = result
        print(f"  ✓ F1 Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        return result
    
    def train_all(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        X_test: sparse.csr_matrix,
        y_test: np.ndarray,
        models: list = None
    ) -> Dict:
        """
        Train all specified baseline models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            models: List of model names to train (default: all)
        
        Returns:
            Dictionary of results
        """
        
        if models is None:
            models = ['logistic_regression', 'random_forest', 'mlp', 'xgboost', 'svm']
        
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)
        
        for model_name in models:
            if model_name == 'logistic_regression':
                self.train_logistic_regression(X_train, y_train, X_test, y_test)
            elif model_name == 'random_forest':
                self.train_random_forest(X_train, y_train, X_test, y_test)
            elif model_name == 'mlp':
                self.train_mlp(X_train, y_train, X_test, y_test)
            elif model_name == 'xgboost':
                self.train_xgboost(X_train, y_train, X_test, y_test)
            elif model_name == 'svm':
                self.train_svm(X_train, y_train, X_test, y_test)
        
        return self.results
    
    def get_summary(self) -> str:
        """Print summary of baseline results"""
        
        summary = "\n" + "="*70 + "\n"
        summary += "BASELINE MODEL COMPARISON\n"
        summary += "="*70 + "\n"
        summary += f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time (s)':<10}\n"
        summary += "-"*70 + "\n"
        
        best_f1 = -1
        best_model = None
        
        for name, result in self.results.items():
            summary += (f"{name:<20} {result['precision']:<12.4f} "
                       f"{result['recall']:<12.4f} {result['f1']:<12.4f} "
                       f"{result['training_time']:<10.2f}\n")
            
            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_model = name
        
        summary += "-"*70 + "\n"
        summary += f"Best Model: {best_model} (F1: {best_f1:.4f})\n"
        summary += "="*70 + "\n"
        
        return summary
    
    def print_summary(self):
        """Print summary"""
        print(self.get_summary())


if __name__ == "__main__":
    print("Baseline models module imported successfully")
