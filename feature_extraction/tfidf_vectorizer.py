from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfVectorizerWrapper:
    def __init__(self, ngram_range=(1, 1), max_features=75000):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            ngram_range (tuple): Range of n-grams to consider
            max_features (int): Maximum number of features to keep
        """
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        
    def fit_transform(self, data):
        """
        Fit and transform the input data.
        
        Args:
            data (iterable): List of text documents
            
        Returns:
            tuple: (TF-IDF matrix, fitted vectorizer)
        """
        tfidf_matrix = self.vectorizer.fit_transform(data).toarray()
        print(f"TF-IDF Vectorization completed with {tfidf_matrix.shape[1]} features")
        return tfidf_matrix, self.vectorizer
    
    def get_feature_names(self):
        """Get the feature names from the vectorizer."""
        return self.vectorizer.get_feature_names_out()