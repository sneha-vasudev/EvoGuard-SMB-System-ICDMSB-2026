"""
Feature Extraction Module
Extracts TF-IDF and behavioral features from email text
Author: Sneha Vasudev
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from typing import Tuple, Any
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
    BEHAVIORAL_FEATURES
)


class FeatureExtractor:
    """Extract TF-IDF and behavioral features from emails"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_names = None
        
    def fit_tfidf(self, texts: pd.Series) -> 'FeatureExtractor':
        """Fit TF-IDF vectorizer on training texts"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.tfidf_vectorizer.fit(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"✓ TF-IDF fitted: {len(self.feature_names)} features")
        return self
    
    def extract_tfidf(self, texts: pd.Series) -> sparse.csr_matrix:
        """Extract TF-IDF features"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        return self.tfidf_vectorizer.transform(texts)
    
    @staticmethod
    def extract_behavioral_features(df: pd.DataFrame) -> np.ndarray:
        """
        Extract domain-specific behavioral features from email data
        
        Features:
        1. message_length: Number of words
        2. num_links: Count of URL patterns
        3. has_html: Presence of HTML tags
        4. has_urgency: Urgency keywords (urgent, asap, deadline, etc.)
        5. free_domain: Free email domain indicators (gmail, yahoo)
        6. has_promo: Promotional indicators (buy, sale, discount, offer)
        7. has_cta: Call-to-action phrases (click here, verify, confirm)
        """
        
        features = pd.DataFrame()
        
        # 1. Message length
        features['message_length'] = df['text'].str.split().str.len().fillna(0).astype(int)
        
        # 2. Number of links (already cleaned to 'URL')
        features['num_links'] = df['text'].str.count('URL').fillna(0).astype(int)
        
        # 3. HTML content
        features['has_html'] = df['text'].str.contains(
            r'<[a-z]', regex=True, na=False
        ).astype(int)
        
        # 4. Urgency keywords
        features['has_urgency'] = df['text'].str.contains(
            r'urgent|asap|immediately|deadline|act now|within.*hours?|time limited',
            case=False, regex=True, na=False
        ).astype(int)
        
        # 5. Free email domains (phishing indicator)
        features['free_domain'] = df['text'].str.contains(
            r'gmail|yahoo|hotmail|outlook|aol|protonmail',
            case=False, regex=True, na=False
        ).astype(int)
        
        # 6. Promotional content
        features['has_promo'] = df['text'].str.contains(
            r'buy|sale|discount|offer|limited time|save|free|coupon|deal|price',
            case=False, regex=True, na=False
        ).astype(int)
        
        # 7. Call-to-action
        features['has_cta'] = df['text'].str.contains(
            r'click here|click link|verify|confirm|update|login|download|submit|register',
            case=False, regex=True, na=False
        ).astype(int)
        
        # Normalize by message length to avoid bias towards longer emails
        for col in features.columns:
            if col != 'message_length':
                features[col] = features[col].astype(float)
        
        return features.values


class FeatureEngineer:
    """Combined feature extraction pipeline"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.scaler_behavioral = None
        
    def fit(self, df: pd.DataFrame):
        """Fit all feature extractors"""
        print("\n[Feature Extraction Pipeline]")
        print("Step 1: Fitting TF-IDF...")
        self.extractor.fit_tfidf(df['text'])
        
        print("Step 2: Computing behavioral features...")
        self.behavioral_features = self.extractor.extract_behavioral_features(df)
        print(f"✓ {self.behavioral_features.shape[1]} behavioral features extracted")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Extract all features and combine TF-IDF with behavioral features
        
        Returns:
            - Combined sparse matrix (TF-IDF + behavioral)
            - Shape: (n_samples, 5007) where 5000 is TF-IDF + 7 behavioral
        """
        # Extract TF-IDF
        tfidf_features = self.extractor.extract_tfidf(df['text'])
        
        # Extract behavioral
        behavioral = self.extractor.extract_behavioral_features(df)
        
        # Convert behavioral to sparse
        behavioral_sparse = sparse.csr_matrix(behavioral.astype(np.float32))
        
        # Combine: TF-IDF + behavioral
        combined = sparse.hstack([tfidf_features, behavioral_sparse])
        
        print(f"✓ Combined features: {combined.shape}")
        print(f"  - TF-IDF: {tfidf_features.shape[1]}")
        print(f"  - Behavioral: {behavioral_sparse.shape[1]}")
        
        return combined
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Fit and transform in one call"""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> list:
        """Get names of all features"""
        tfidf_names = list(self.extractor.feature_names)
        behavioral_names = BEHAVIORAL_FEATURES
        return tfidf_names + behavioral_names
    
    def get_feature_count(self) -> int:
        """Total number of features"""
        return len(self.extractor.feature_names) + len(BEHAVIORAL_FEATURES)


def prepare_features(
    X_train_text: pd.Series,
    X_test_text: pd.Series,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Complete feature preparation pipeline
    
    Args:
        X_train_text: Training email texts
        X_test_text: Test email texts
        df_train: Training dataframe (for behavioral features)
        df_test: Test dataframe (for behavioral features)
    
    Returns:
        X_train_features: Sparse feature matrix for training
        X_test_features: Sparse feature matrix for testing
    """
    
    engineer = FeatureEngineer()
    
    # Fit on training data
    engineer.fit(df_train)
    
    # Transform both train and test
    X_train_features = engineer.transform(df_train)
    X_test_features = engineer.transform(df_test)
    
    return X_train_features, X_test_features, engineer


if __name__ == "__main__":
    # Test feature extraction
    from data_loader import load_and_prepare_data
    
    print("Testing Feature Extraction...")
    df, mappings = load_and_prepare_data()
    
    engineer = FeatureEngineer()
    engineer.fit(df.head(100))
    X = engineer.transform(df.head(100))
    
    print(f"\nExtracted features shape: {X.shape}")
    print(f"Feature names ({len(engineer.get_feature_names())} total):")
    print(engineer.get_feature_names()[-10:])  # Last 10 (behavioral)
