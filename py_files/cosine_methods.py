import numpy as np
from sklearn.neighbors import NearestNeighbors

# Compares embeddings using cosine similarity with a nearest neighbor approach
def compare_embeddings_cosine_nearest_neighbor(embeddings):
    num_embeddings = len(embeddings)
    similarity_matrix = np.zeros((num_embeddings, num_embeddings))

    # Convert embeddings to uint8
    embeddings_uint8 = np.array([np.array(embedding * 255, dtype=np.uint8) for embedding in embeddings])

    # Initialize Nearest Neighbors model with n_neighbors=10
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(embeddings_uint8)

    for i in range(num_embeddings):
        distances, indices = nbrs.kneighbors([embeddings_uint8[i]], n_neighbors=10)
        
        for dist, idx in zip(distances[0], indices[0]):
            similarity_matrix[i, idx] = 1 - dist

    return similarity_matrix
