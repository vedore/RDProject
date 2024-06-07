import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_lexical_similarity(first_classes_list, second_classes_list):

    # Fits the constructor with all the cleared data
    # print("Fitting")
    vectorizer = TfidfVectorizer().fit(first_classes_list + second_classes_list)

    # Split the TF-IDF matrix back into the two original lists
    tfidf_matrix_first_labels = vectorizer.transform(first_classes_list)

    tfidf_matrix_second_labels = vectorizer.transform(second_classes_list)

    # Compute the cosine similarity between the two lists
    similarity_matrix = cosine_similarity(tfidf_matrix_first_labels, tfidf_matrix_second_labels)

    # Create a DataFrame for better visualization
    # similarity_df = pd.DataFrame(similarity_matrix, index=first_labels_lower, columns=second_labels_lower)

    return similarity_matrix


