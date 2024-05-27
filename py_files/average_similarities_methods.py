def compute_average_similarities(jaccard_matrix, cosine_matrix):
    combined_scores = []

    for jaccard, cosine in zip(jaccard_matrix, cosine_matrix):
        assert jaccard["IRI:source"] == cosine["IRI:source"], "IRIs must match for averaging"
        assert jaccard["label:source"] == cosine["label:source"], "Labels must match for averaging"

        avg_similarity = (jaccard["jaccard_score"] + cosine["cosine_score"]) / 2
        combined_scores.append({
            "IRI:source": jaccard["IRI:source"],
            "label:source": jaccard["label:source"],
            "average_similarity": avg_similarity
        })

    return combined_scores