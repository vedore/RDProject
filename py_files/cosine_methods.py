import numpy as np
from sklearn.neighbors import NearestNeighbors

# Compares embeddings using cosine similarity with a nearest neighbor approach
def compare_embeddings_cosine_nearest_neighbor(embeddings):
    similarity_scores = []

    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(embeddings)

    for i in range(len(embeddings)):
        # Find nearest neighbors for each embedding
        distances, indices = nbrs.kneighbors([embeddings[i]])
        
        # Compute average cosine similarity with nearest neighbors
        avg_similarity = np.mean(1 - distances)
        similarity_scores.append(avg_similarity)

    return similarity_scores