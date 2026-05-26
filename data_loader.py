"""
Data Loading and Preprocessing Module
Handles loading multiple email datasets and preparing them for training
Author: Sneha Vasudev
"""

import pandas as pd
import numpy as np
import re
import emoji
import warnings
from pathlib import Path
from typing import Tuple, Dict
from config import DATASET_PATHS, BEHAVIORAL_FEATURES

warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess multiple email datasets"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean email text by removing URLs, emoji, special characters"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = emoji.demojize(text)
        text = re.sub(r'http\S+', ' URL ', text)
        text = re.sub(r'[^a-z0-9_: ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def load_enron_dataset(path: Path) -> pd.DataFrame:
        """Load Enron email corpus"""
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df[['text']].dropna()
        df['is_spam'] = 0
        df['is_phishing'] = 0
        df['sentiment'] = 'neutral'
        df['y_intent'] = 'informational'
        df['urgent'] = 0
        return df
    
    @staticmethod
    def load_trec_dataset(path: Path) -> pd.DataFrame:
        """Load TREC spam corpus"""
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df.rename(columns={df.columns[0]: 'is_spam', df.columns[1]: 'text'})
        df['is_phishing'] = 0
        df['sentiment'] = 'neutral'
        df['y_intent'] = 'informational'
        df['urgent'] = 0
        return df[['text', 'is_spam', 'is_phishing', 'sentiment', 'y_intent', 'urgent']]
    
    @staticmethod
    def load_nazario_dataset(path: Path) -> pd.DataFrame:
        """Load Nazario phishing corpus"""
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df[['text']].dropna()
        df['is_spam'] = 0
        df['is_phishing'] = 1
        df['sentiment'] = 'neutral'
        df['y_intent'] = 'informational'
        df['urgent'] = 1
        return df
    
    @staticmethod
    def load_sentiment_dataset(path: Path) -> pd.DataFrame:
        """Load sentiment-annotated dataset (Sentiment140, GoEmotions)"""
        df = pd.read_csv(path, on_bad_lines='skip')
        if 'sentiment' not in df.columns:
            df['sentiment'] = 'neutral'
        if 'text' not in df.columns and len(df.columns) > 0:
            df = df.rename(columns={df.columns[0]: 'text'})
        
        df['is_spam'] = 0
        df['is_phishing'] = 0
        df['y_intent'] = 'informational'
        df['urgent'] = 0
        return df[['text', 'is_spam', 'is_phishing', 'sentiment', 'y_intent', 'urgent']]
    
    @classmethod
    def load_all_datasets(cls) -> pd.DataFrame:
        """Load and combine all available datasets"""
        dataframes = []
        
        dataset_loaders = {
            'enron': cls.load_enron_dataset,
            'trec07': cls.load_trec_dataset,
            'nazario': cls.load_nazario_dataset,
            'complaints': cls.load_sentiment_dataset,
            'bitext': cls.load_sentiment_dataset,
            'email_meta': cls.load_sentiment_dataset,
        }
        
        for name, loader in dataset_loaders.items():
            path = DATASET_PATHS.get(name)
            if path and path.exists():
                try:
                    df = loader(path)
                    if len(df) > 0:
                        print(f"✓ Loaded {name}: {len(df)} samples")
                        dataframes.append(df)
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
        
        if not dataframes:
            raise ValueError("No datasets loaded. Check DATASET_PATHS.")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        print(f"\n✓ Combined dataset: {len(combined_df)} unique samples")
        return combined_df
    
    @staticmethod
    def prepare_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize and encode labels for multi-task learning
        Returns dataframe and label mappings
        """
        label_mappings = {}
        
        # Threat labels (0=safe, 1=spam, 2=phishing)
        def encode_threat(row):
            if row['is_phishing'] == 1:
                return 2
            elif row['is_spam'] == 1:
                return 1
            return 0
        
        df['threat'] = df.apply(encode_threat, axis=1)
        label_mappings['threat'] = {0: 'safe', 1: 'spam', 2: 'phishing'}
        
        # Sentiment labels (0=negative, 1=neutral, 2=positive)
        sentiment_map = {
            'negative': 0, 'neg': 0, '-1': 0, 0: 0,
            'neutral': 1, 'none': 1, '0': 1,
            'positive': 2, 'pos': 2, '1': 2, '4': 2
        }
        df['sentiment'] = df['sentiment'].map(sentiment_map).fillna(1).astype(int)
        label_mappings['sentiment'] = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Intent labels (0=informational, 1=complaint, 2=purchase)
        intent_map = {
            'informational': 0, 'info': 0, 'question': 0,
            'complaint': 1, 'complain': 1, 'issue': 1,
            'purchase': 2, 'buy': 2, 'order': 2
        }
        df['intent'] = df['y_intent'].map(intent_map).fillna(0).astype(int)
        label_mappings['intent'] = {0: 'informational', 1: 'complaint', 2: 'purchase'}
        
        # Urgency labels (0=normal, 1=high)
        df['urgency'] = (df['urgent'].astype(int) | 
                        df['threat'].isin([1, 2]).astype(int)).astype(int)
        label_mappings['urgency'] = {0: 'normal', 1: 'high'}
        
        return df, label_mappings
    
    @classmethod
    def prepare_data(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Main preparation pipeline"""
        print("\n[1/3] Cleaning text...")
        df['text'] = df['text'].apply(cls.clean_text)
        df = df[df['text'].str.len() > 5].reset_index(drop=True)
        
        print("[2/3] Standardizing labels...")
        df, label_mappings = cls.prepare_labels(df)
        
        print("[3/3] Data preparation complete!")
        print(f"  Final dataset: {len(df)} samples")
        
        return df, label_mappings


def load_and_prepare_data() -> Tuple[pd.DataFrame, Dict]:
    """Load all datasets and prepare for training"""
    loader = DataLoader()
    df = loader.load_all_datasets()
    df, label_mappings = loader.prepare_data(df)
    return df, label_mappings


if __name__ == "__main__":
    # Test data loading
    print("Testing DataLoader...")
    df, mappings = load_and_prepare_data()
    print("\nLabel mappings:")
    for task, mapping in mappings.items():
        print(f"  {task}: {mapping}")
    print("\nSample data:")
    print(df[['text', 'threat', 'sentiment', 'intent', 'urgency']].head())
