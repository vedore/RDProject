def combine_similarities(lexical_similarity, semantic_similarity, alpha=0.5):
    combined_similarity = alpha * lexical_similarity + (1 - alpha) * semantic_similarity
    return combined_similarity
