import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_lexical_similarity(first_classes_list, second_classes_list):

    # Get all the labels passed them to string and join, lower them
    # print("First Label")
    first_labels_lower = [' '.join(str(label).lower() for label in labels) for iri, labels in first_classes_list]

    # print("Second Label")
    second_labels_lower = [' '.join(str(label).lower() for label in labels) for iri, labels in second_classes_list]

    # print(first_labels_lower_flat)

    # Fits the constructor with all the cleared data
    # print("Fitting")
    vectorizer = TfidfVectorizer().fit(first_labels_lower + second_labels_lower)

    # Split the TF-IDF matrix back into the two original lists
    tfidf_matrix_first_labels = vectorizer.transform(first_labels_lower)

    tfidf_matrix_second_labels = vectorizer.transform(second_labels_lower)

    # Compute the cosine similarity between the two lists
    similarity_matrix = cosine_similarity(tfidf_matrix_first_labels, tfidf_matrix_second_labels)

    # Create a DataFrame for better visualization
    similarity_df = pd.DataFrame(similarity_matrix, index=first_labels_lower, columns=second_labels_lower)

    print(similarity_df)

    return similarity_matrix


