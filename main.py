"""
EvoGuard-SMB: Main Training and Evaluation Script
Autonomous Neuroevolutionary Email Intelligence Agent
Author: Sneha Vasudev
"""

import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import modules
from config import (
    TEST_SIZE, RANDOM_STATE, BASELINE_SAMPLE_SIZE, OUTPUT_DIR,
    TASK_NAMES, TASK_SIZES
)
from data_loader import load_and_prepare_data
from feature_extraction import prepare_features, FeatureEngineer
from baselines import BaselineModels
from neat_classifier import NEATClassifier
from agent import EmailAgent


def main(args):
    """Main pipeline execution"""
    
    print("\n" + "="*80)
    print("EVOGUARD-SMB: AUTONOMOUS NEUROEVOLUTIONARY EMAIL INTELLIGENCE AGENT")
    print("="*80)
    
    # ========== DATA LOADING ==========
    print("\n[STAGE 1: DATA LOADING AND PREPARATION]")
    print("-"*80)
    
    df, label_mappings = load_and_prepare_data()
    
    # ========== FEATURE EXTRACTION ==========
    print("\n[STAGE 2: FEATURE EXTRACTION]")
    print("-"*80)
    
    engineer = FeatureEngineer()
    X_features = engineer.fit_transform(df)
    
    # Multi-task labels
    Y = df[['threat', 'sentiment', 'intent', 'urgency']].values
    
    print(f"✓ Total features: {X_features.shape[1]}")
    print(f"✓ Samples: {X_features.shape[0]}")
    print(f"✓ Label shape: {Y.shape}")
    
    # ========== TRAIN-TEST SPLIT ==========
    print("\n[STAGE 3: TRAIN-TEST SPLIT]")
    print("-"*80)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_features, Y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y[:, 0]  # Stratify by threat label
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # For baselines, use stratified sample
    if X_train.shape[0] > BASELINE_SAMPLE_SIZE:
        indices = np.random.choice(X_train.shape[0], BASELINE_SAMPLE_SIZE, replace=False)
        X_train_baseline = X_train[indices]
        Y_train_baseline = Y_train[indices]
        print(f"✓ Baseline sample: {BASELINE_SAMPLE_SIZE} samples")
    else:
        X_train_baseline = X_train
        Y_train_baseline = Y_train
    
    # ========== THREAT DETECTION (PRIMARY TASK) ==========
    print("\n[STAGE 4: BASELINE MODELS - THREAT DETECTION]")
    print("-"*80)
    
    baselines = BaselineModels()
    y_train_threat = Y_train_baseline[:, 0]
    y_test_threat = Y_test[:, 0]
    
    baseline_results = baselines.train_all(
        X_train_baseline, y_train_threat,
        X_test, y_test_threat,
        models=['logistic_regression', 'random_forest', 'mlp', 'xgboost', 'svm']
    )
    
    baselines.print_summary()
    
    # ========== NEAT MULTI-TASK MODEL ==========
    if args.train_neat:
        print("\n[STAGE 5: NEAT CLASSIFIER - MULTI-TASK LEARNING]")
        print("-"*80)
        
        # Convert to dense for NEAT
        print("Converting features to dense format...")
        X_train_dense = X_train.toarray().astype(np.float32)
        X_test_dense = X_test.toarray().astype(np.float32)
        
        # Normalize features
        X_mean = X_train_dense.mean(axis=0)
        X_std = X_train_dense.std(axis=0) + 1e-8
        X_train_dense = (X_train_dense - X_mean) / X_std
        X_test_dense = (X_test_dense - X_mean) / X_std
        
        print(f"✓ Dense features: {X_train_dense.shape}")
        
        # Train NEAT
        neat_model = NEATClassifier(
            n_inputs=X_train_dense.shape[1],
            task_sizes=TASK_SIZES
        )
        
        t0 = time.time()
        results = neat_model.train(
            X_train_dense, Y_train,
            generations=args.generations,
            verbose=True
        )
        neat_time = time.time() - t0
        
        # Evaluate NEAT
        print("\n[STAGE 6: NEAT EVALUATION]")
        print("-"*80)
        
        predictions, probabilities = neat_model.predict(X_test_dense)
        
        from sklearn.metrics import f1_score
        
        print("\nPer-Task Performance:")
        print("-"*80)
        
        task_f1_scores = []
        for task_idx, task_name in enumerate(TASK_NAMES):
            f1 = f1_score(Y_test[:, task_idx], predictions[:, task_idx],
                         average='weighted', zero_division=0)
            task_f1_scores.append(f1)
            print(f"  {task_name:<15} F1: {f1:.4f}")
        
        weighted_avg = np.average(task_f1_scores, weights=[0.40, 0.30, 0.20, 0.10])
        print(f"  {'Weighted Avg':<15} F1: {weighted_avg:.4f}")
        
        # Save model
        model_path = OUTPUT_DIR / "neat_model.pkl"
        neat_model.save(str(model_path))
        print(f"\n✓ Model saved to {model_path}")
        
        neat_results = {
            'model': neat_model,
            'predictions': predictions,
            'probabilities': probabilities,
            'weighted_f1': weighted_avg,
            'training_time': neat_time,
            'task_f1_scores': dict(zip(TASK_NAMES, task_f1_scores))
        }
    else:
        neat_results = None
    
    # ========== AUTONOMOUS AGENT ==========
    print("\n[STAGE 7: AUTONOMOUS AGENT DEPLOYMENT]")
    print("-"*80)
    
    if neat_results:
        agent = EmailAgent()
        
        # Process test emails
        for i in range(min(100, len(X_test))):
            email_id = f"email_{i:04d}"
            email_text = df.iloc[i]['text'][:500]
            
            pred_dict = {
                'threat': predictions[i, 0],
                'sentiment': predictions[i, 1],
                'intent': predictions[i, 2],
                'urgency': predictions[i, 3]
            }
            
            prob_dict = {
                'threat': probabilities[i, :3],
                'sentiment': probabilities[i, 3:6],
                'intent': probabilities[i, 6:9],
                'urgency': probabilities[i, 9:]
            }
            
            agent.process_email(email_id, email_text, pred_dict, prob_dict)
        
        # Generate report
        report_path = agent.export_summary_to_file()
        print(f"✓ Summary report exported to {report_path}")
        
        metrics = agent.get_performance_metrics()
        print(f"\nAgent Performance Metrics:")
        print("-"*80)
        for metric, value in metrics.items():
            print(f"  {metric:<25} {value}")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    if neat_results:
        print(f"\nBest Baseline Model: {max(baseline_results.items(), key=lambda x: x[1]['f1'])[0]}")
        print(f"  F1 Score: {max([r['f1'] for r in baseline_results.values()]):.4f}")
        
        print(f"\nEvoGuard-SMB (NEAT) Performance:")
        print(f"  Weighted F1: {neat_results['weighted_f1']:.4f}")
        print(f"  Training Time: {neat_results['training_time']:.2f}s")
        print(f"  Per-task scores:")
        for task_name, f1 in neat_results['task_f1_scores'].items():
            print(f"    - {task_name}: {f1:.4f}")
    
    print("\n✓ Pipeline execution complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EvoGuard-SMB: Email Intelligence Agent Training"
    )
    
    parser.add_argument(
        '--train-neat',
        action='store_true',
        default=True,
        help='Train NEAT model (default: True)'
    )
    
    parser.add_argument(
        '--no-neat',
        dest='train_neat',
        action='store_false',
        help='Skip NEAT training'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of NEAT generations (default: 50)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
