import numpy as np
from sklearn.neighbors import NearestNeighbors

# Compares embeddings using cosine similarity with a nearest neighbor approach
def compare_embeddings_cosine_nearest_neighbor(embeddings):
    similarity_scores = []

    # Convert embeddings to uint8
    embeddings_uint8 = [np.array(embedding * 255, dtype=np.uint8) for embedding in embeddings]

    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(embeddings_uint8)

    for i in range(len(embeddings_uint8)):
        # Find nearest neighbors for each embedding
        distances, indices = nbrs.kneighbors([embeddings_uint8[i]])
        
        # Compute average cosine similarity with nearest neighbors
        avg_similarity = np.mean(1 - distances)
        similarity_scores.append(avg_similarity)

    return similarity_scores