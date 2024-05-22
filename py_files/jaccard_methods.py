import numpy as np

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
    num_labels = len(labels)
    similarity_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                similarity_matrix[i][j] = 1.0  # A label is always similar to itself
            else:
                similarity_matrix[i][j] = calculate_jaccard_similarity_between_labels(labels[i], labels[j])

    return similarity_matrix