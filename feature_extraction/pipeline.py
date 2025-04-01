from .word_embeddings import WordEmbeddings
from .tfidf_vectorizer import TfidfVectorizerWrapper

class FeatureEngineer:
    def __init__(self, method='word2vec'):
        self.method = method
        self.vectorizer = None

    def fit_transform(self, processed_df):
        """Input: DataFrame from data_processing"""
        texts = processed_df['normalized'].tolist()
        
        if self.method == 'word2vec':
            model = WordEmbeddings()
            model.train(texts)
            vectors = model.get_vectors(texts)
        else:
            tfidf = TfidfVectorizerWrapper()
            vectors, self.vectorizer = tfidf.fit_transform(texts)
            
        return {
            'vectors': vectors,
            'original_data': processed_df
        }