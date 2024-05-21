def compute_average_similarities(human_results, mouse_results):
    combined_scores = []

    for human, mouse in zip(human_results, mouse_results):
        assert human["IRI:source"] == mouse["IRI:source"], "IRIs must match for averaging"
        assert human["label:source"] == mouse["label:source"], "Labels must match for averaging"

        avg_similarity = (human["jaccard_score"] + mouse["cosine_score"]) / 2
        combined_scores.append({
            "IRI:source": human["IRI:source"],
            "label:source": human["label:source"],
            "average_similarity": avg_similarity
        })

    return combined_scores