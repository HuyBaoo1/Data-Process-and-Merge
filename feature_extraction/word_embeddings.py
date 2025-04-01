import gensim
import numpy as np
import logging
from gensim.models import KeyedVectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class WordEmbeddings:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Initialize word embeddings processor.
        
        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Maximum distance between current and predicted word
            min_count (int): Ignores words with frequency lower than this
            workers (int): Number of worker threads
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def load_pretrained(self, word2vec_path, limit=200000):
        """
        Load pretrained Word2Vec embeddings.
        
        Args:
            word2vec_path (str): Path to pretrained model
            limit (int): Limit number of vectors to load
            
        Returns:
            Loaded model
        """
        logging.info(f"Loading pretrained Word2Vec model from {word2vec_path}...")
        self.model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=limit)
        logging.info("Pretrained Word2Vec model loaded successfully.")
        return self.model
        
    def train(self, data):
        """
        Train a custom Word2Vec model.
        
        Args:
            data (iterable): List of text documents
            
        Returns:
            Trained model
        """
        tokenized_data = [sentence.split() for sentence in data]
        self.model = gensim.models.Word2Vec(
            sentences=tokenized_data,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        logging.info("Custom Word2Vec model trained successfully.")
        return self.model
        
    def get_vectors(self, data):
        """
        Generate word vectors for input data.
        
        Args:
            data (iterable): List of text documents
            
        Returns:
            numpy.ndarray: Array of word vectors
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load or train a model first.")
            
        word_vectors = []
        for sentence in data:
            words = sentence.split()
            vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            vector = np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
            word_vectors.append(vector)
        return np.array(word_vectors)