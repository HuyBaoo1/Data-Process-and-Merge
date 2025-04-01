import pandas as pd
import numpy as np
from preprocessing import normalize_text, remove_frequent_rare_words
from keyword_grouping import group_keywords
#from umap import UMAP
#from sklearn.cluster import KMeans, DBSCAN
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from data_processing.init import load_data, save_filtered_data
from feature_extraction.frequency_vector import FrequencyVectorizer
from feature_extraction.tfidf_vectorizer import TfidfVectorizerWrapper
from feature_extraction.word_embeddings import WordEmbeddings

def main():
    # File path
    file_path = "C:/Users/ASUS/Downloads/Gửi Huy.xlsx"

    # 1. Load and preprocess data
    df = load_data(file_path)
    
    # 2. Text normalization
    df['normalized_keywords'] = df['keyword'].astype(str).apply(normalize_text)
    
    # 3. Filter spam patterns
    df = df[~df['keyword'].astype(str).str.match(r'^\d{9,}$', na=False)]
    df = df[~df['keyword'].astype(str).str.match(r'(.)\1{3,}', na=False)]

    # 4. Remove frequent/rare words
    word_freq = pd.Series(' '.join(df['normalized_keywords']).split()).value_counts()
    frequent_words = set(word_freq[word_freq > 0.95 * len(df)].index)
    rare_words = set(word_freq[word_freq < 5].index)
    df['normalized_keywords'] = df['normalized_keywords'].apply(
        lambda text: remove_frequent_rare_words(text, frequent_words, rare_words)
    )

    # 5. Feature extraction (NEW)
    # Choose one method or implement a hybrid approach:
    
    # Method 1: TF-IDF
    tfidf = TfidfVectorizerWrapper()
    tfidf_vectors, _ = tfidf.fit_transform(df['normalized_keywords'])
    
    # Method 2: Word Embeddings (recommended for semantic similarity)
    we = WordEmbeddings()
    we.train(df['normalized_keywords'].tolist())
    embedding_vectors = we.get_vectors(df['normalized_keywords'])
    
    # Add vectors to DataFrame
    df['feature_vectors'] = list(embedding_vectors)  # Using word2vec embeddings

    # 6. Keyword grouping (modified to use embeddings)
    def enhanced_grouping(keywords, vectors):
        """Group keywords using both text similarity and vector similarity"""
        keyword_map = {}
        
        # First pass: traditional text similarity
        text_groups = group_keywords(keywords.unique())
        
        # Second pass: refine with vector similarity
        for main_word, variants in text_groups.items():
            main_vec = vectors[keywords == main_word][0]
            for variant in variants:
                if np.dot(main_vec, vectors[keywords == variant][0]) > 0.7:  # similarity threshold
                    keyword_map[variant] = main_word
                else:
                    keyword_map[variant] = variant
        return keyword_map

    keyword_map = enhanced_grouping(df['normalized_keywords'], np.array(df['feature_vectors'].tolist()))
    df["merged_keywords"] = df["normalized_keywords"].map(keyword_map)

    # 7. Aggregation and filtering (existing logic)
    df_grouped = df.groupby("merged_keywords").agg({
        "Searched Count": "sum",
        "Search-to-watch": "sum",
    }).reset_index()

    df_grouped['CTR'] = (df_grouped['Search-to-watch'] / df_grouped['Searched Count'].replace(0, np.nan)) * 100
    df_grouped['CTR'] = df_grouped['CTR'].fillna(0).round(2)

    # Set thresholds
    MIN_SEARCH_VOLUME = 10
    MIN_CTR = 10

    filtered_df = df_grouped[
        (df_grouped['Searched Count'] >= MIN_SEARCH_VOLUME) & 
        (df_grouped['CTR'] >= MIN_CTR)
    ]

    # 8. Save results
    save_filtered_data(filtered_df, "filtered_data_with_embeddings.csv", file_path, sheet_name="EnhancedFilteredData")

    print(f"Enhanced dataset saved with {len(filtered_df)} keywords.")
    print(f"Vector dimensions: {embedding_vectors.shape[1]}")

if __name__ == "__main__":
    main()