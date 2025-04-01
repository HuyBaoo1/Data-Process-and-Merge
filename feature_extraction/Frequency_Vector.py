import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class FrequencyVectorizer:
    def __init__(self, ngram_range=(1, 1), max_features=75000):
        """
        Initialize the frequency vectorizer.
        
        Args:
            ngram_range (tuple): Range of n-grams to consider
            max_features (int): Maximum number of features to keep
        """
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        
    def fit_transform(self, data):
        """
        Fit and transform the input data.
        
        Args:
            data (iterable): List of text documents
            
        Returns:
            tuple: (array of vectors, fitted vectorizer)
        """
        emb = self.vectorizer.fit_transform(data).toarray()
        print(f"Count vectorized with {emb.shape[1]} features")
        return emb, self.vectorizer
    
    def analyze_features(self, emb, vectorizer, ngram_label="Uni-gram", sample_index=0, sample_data=None):
        """
        Print analysis of the vectorized features.
        
        Args:
            emb: Vectorized output
            vectorizer: Fitted vectorizer
            ngram_label (str): Label for the n-gram type
            sample_index (int): Index of sample to display
            sample_data: Original text data (optional)
        """
        print(f"\n{ngram_label} bag-of-words analysis:")
        print("-" * 50)
        print("Feature names:", vectorizer.get_feature_names_out(), "\n")
        print("Vocabulary:", vectorizer.vocabulary_, "\n")
        
        if sample_data is not None and len(sample_data) > sample_index:
            print("Sample text:", sample_data[sample_index])
            print("Vector representation:", emb[sample_index], "\n")