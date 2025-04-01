from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

def cluster_keywords(vectors):
    # Thử nghiệm 2 phương pháp
    # Phương án 1: DBSCAN (tự động xác định số cụm)
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(vectors)
    
    # Phương án 2: K-Means (nếu biết trước số cụm)
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(vectors)
    
    # Chọn phương án tốt hơn qua silhouette score
    if len(set(dbscan_labels)) > 1:
        dbscan_score = silhouette_score(vectors, dbscan_labels)
    else:
        dbscan_score = -1
        
    kmeans_score = silhouette_score(vectors, kmeans_labels)
    
    return (dbscan_labels if dbscan_score > kmeans_score 
            else kmeans_labels)