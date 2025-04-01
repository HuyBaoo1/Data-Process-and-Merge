from sklearn.manifold import TSNE
from umap import UMAP

def reduce_dimensions(vectors):
    # Giảm chiều bằng UMAP (tốt hơn t-SNE với dữ liệu lớn)
    reducer = UMAP(n_components=2, random_state=42)
    reduced_vectors = reducer.fit_transform(vectors)
    return reduced_vectors