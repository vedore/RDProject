# Computes the Jaccard similarity between two labels
def calculate_jaccard_similarity_between_labels(label1, label2):

    # Tokenize labels into sets of words
    set_label_1 = set(label1.lower().split())
    set_label_2 = set(label2.lower().split())

    # This line calculates the size of the intersection between two sets, set_label_1 and set_label_2
    intersection = len(set_label_1.intersection(set_label_2))

    # Calculates the size of the union of two sets, set_label_1 and set_label_2.
    union = len(set_label_1.union(set_label_2))

    #Finally, Calculates The Similarity
    jaccard_similarity = intersection / union

    return jaccard_similarity

# Computes the Jaccard similarity scores between all pairs of labels in a list
def compare_labels_jaccard(labels):
    similarity_scores = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            similarity = calculate_jaccard_similarity_between_labels(labels[i], labels[j])
            similarity_scores.append(similarity)

    return similarity_scores