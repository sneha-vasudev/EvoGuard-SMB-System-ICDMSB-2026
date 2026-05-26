"""
Configuration settings for EvoGuard-SMB Email Intelligence Agent
Author: Sneha Vasudev
Date: 2026
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "database"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AGENT_DIR = PROJECT_ROOT / "agent"
MODEL_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, AGENT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============= DATA CONFIGURATION =============
DATASET_PATHS = {
    "enron": DATA_DIR / "Enron.csv",
    "trec07": DATA_DIR / "TREC-07.csv",
    "nazario": DATA_DIR / "Nazario_5.csv",
    "complaints": DATA_DIR / "complaints_processed.csv",
    "bitext": DATA_DIR / "Bitext_Sample_Customer_Service_Training.csv",
    "email_meta": DATA_DIR / "Email Analysis Dataset.csv",
}

# ============= FEATURE EXTRACTION =============
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Behavioral features
BEHAVIORAL_FEATURES = [
    'message_length',
    'num_links',
    'has_html',
    'has_urgency',
    'free_domain',
    'has_promo',
    'has_cta'
]

# ============= MULTI-TASK LEARNING =============
TASK_SIZES = [3, 3, 3, 1]  # [Threat (3 classes), Sentiment (3), Intent (3), Urgency (binary)]
TASK_NAMES = ['threat', 'sentiment', 'intent', 'urgency']
TASK_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

THREAT_LABELS = ['safe', 'spam', 'phishing']
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
INTENT_LABELS = ['informational', 'complaint', 'purchase']
URGENCY_LABELS = ['normal', 'high']

# ============= NEAT CONFIGURATION =============
NEAT_CONFIG = {
    'pop_size': 100,
    'generations': 50,
    'fitness_threshold': 0.95,
    'activation_default': 'tanh',
    'activation_mutate_rate': 0.0,
    'activation_options': ['tanh'],
}

# NEAT fitness function weights
FITNESS_WEIGHTS = {
    'f1_score': 0.5,
    'novelty': 0.2,
    'latency_penalty': -0.3
}

# ============= BASELINE MODELS =============
BASELINE_SAMPLE_SIZE = 50000  # Use stratified sample for baselines
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.1

# Hyperparameters for baselines
BASELINE_PARAMS = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': RANDOM_STATE,
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
    'mlp': {
        'hidden_layer_sizes': (256, 128),
        'max_iter': 100,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': VALIDATION_FRACTION,
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'svm': {
        'C': 1.0,
        'max_iter': 5000,
        'random_state': RANDOM_STATE,
    }
}

# ============= INFERENCE =============
INFERENCE_BATCH_SIZE = 32
INFERENCE_TIMEOUT = 5.0  # seconds
MAX_EMAIL_LENGTH = 10000  # characters

# ============= DEPLOYMENT =============
DEPLOYMENT_MEMORY_LIMIT_MB = 512
TARGET_INFERENCE_LATENCY_MS = 0.50  # milliseconds
QUARANTINE_FOLDER = "spam"
PRIORITY_FOLDER = "high_priority"

# ============= LOGGING =============
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = "INFO"

# ============= AGENT SETTINGS =============
AGENT_ENABLED = True
AUTO_QUARANTINE_ENABLED = True
AUTO_PRIORITY_ROUTING_ENABLED = True
CRM_LOGGING_ENABLED = True
SUMMARY_FREQUENCY = "weekly"  # 'daily', 'weekly', 'monthly'
