import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import scipy.sparse as sp


def jaccard_similarity_matrix(matrix1, matrix2):
    # Ensure the input matrices are in CSR format
    if not sp.isspmatrix_csr(matrix1):
        matrix1 = matrix1.tocsr()
    if not sp.isspmatrix_csr(matrix2):
        matrix2 = matrix2.tocsr()

    # Compute the intersection and union
    intersection = matrix1.dot(matrix2.T).tocsc()
    sum_matrix1 = matrix1.sum(axis=1)
    sum_matrix2 = matrix2.sum(axis=1)

    union = sum_matrix1 + sum_matrix2.T - intersection

    # Convert union to a numpy array if it's not already
    if sp.isspmatrix(union):
        union = union.toarray()
    else:
        union = np.array(union)

    # Avoid division by zero
    union[union == 0] = 1

    # Compute Jaccard similarity
    similarity = intersection.toarray() / union

    return similarity


def compute_semantic_similarity(first_labels, second_labels):
    # Create the CountVectorizer to get the bag-of-words representation
    vectorizer = CountVectorizer(binary=True).fit(first_labels + second_labels)

    # Transform both sets of labels using the same CountVectorizer instance
    binary_matrix_first_labels = vectorizer.transform(first_labels)
    binary_matrix_second_labels = vectorizer.transform(second_labels)

    # Compute the Jaccard similarity matrix
    similarities = jaccard_similarity_matrix(binary_matrix_first_labels, binary_matrix_second_labels)

    return similarities


# Computes the Jaccard similarity between two labels
def calculate_jaccard_similarity_between_labels(label1, label2):

    set_label_1 = set(label1.split())
    set_label_2 = set(label2.split())

    # This line calculates the size of the intersection between two sets, set_label_1 and set_label_2
    intersection = len(set_label_1.intersection(set_label_2))

    # Calculates the size of the union of two sets, set_label_1 and set_label_2.
    union = len(set_label_1.union(set_label_2))

    # Finally, Calculates The Similarity
    jaccard_similarity = intersection / union

    return jaccard_similarity


# Computes the Jaccard similarity scores between all pairs of labels in a list
def compute_semantic_similarity_2(first_labels, second_labels):
    first_num_labels = len(first_labels)
    second_num_labels = len(second_labels)

    similarity_matrix = np.zeros((first_num_labels, second_num_labels))

    print(first_num_labels)

    for i in range(first_num_labels):
        print(str(i) + " -> " + first_labels[i])
        for j in range(second_num_labels):
            similarity_matrix[i][j] = calculate_jaccard_similarity_between_labels(first_labels[i], second_labels[j])

    return similarity_matrix
