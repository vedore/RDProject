import numpy as np
import torch
from py_files.cosine_methods import compare_embeddings_cosine_nearest_neighbor
from py_files.embeddings_extractor import create_embedding_file_from_owl_file, get_data_from_npy_file
from py_files.jaccard_methods import compare_labels_jaccard
import pandas as pd


def compute_average_similarities(jaccard_scores, cosine_scores):
    # Ensure the matrices are of the same size
    assert jaccard_scores.shape == cosine_scores.shape, "The similarity matrices must have the same shape."
    
    # Compute the average similarity scores
    avg_similarities = (jaccard_scores + cosine_scores) / 2
    
    return avg_similarities

def compute_average_lexical_similarities(jaccard_scores):
    return sum(jaccard_scores) / len(jaccard_scores)

def compute_average_embedding_similarities(cosine_scores):
    return sum(cosine_scores) / len(cosine_scores)

def project_combined_class(human_iris, human_labels, human_embeddings, mouse_iris, mouse_labels, mouse_embeddings):
    combined_iris = human_iris + mouse_iris
    combined_labels = human_labels + mouse_labels
    combined_embeddings = np.concatenate((human_embeddings, mouse_embeddings), axis=0)

    print("Combined Jaccard Similarity Step")
    combined_jaccard_similarity_matrix = compare_labels_jaccard(combined_labels)

    print("Combined Cosine Similarity Step")
    combined_cosine_similarity_matrix = compare_embeddings_cosine_nearest_neighbor(combined_embeddings)

    return combined_jaccard_similarity_matrix, combined_cosine_similarity_matrix, combined_iris, combined_labels

def display_similarity_matrix(similarity_matrix, labels):
    df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    print(df)

def main():
    print("Human Ontology Process:\n")
    human_iris, human_labels, human_embeddings = get_data_from_npy_file(
        "embeddings/human_embeddings.npy",
        "embeddings/human_iris_labels.npy"
    )

    print("Mouse Ontology Process:\n")
    mouse_iris, mouse_labels, mouse_embeddings = get_data_from_npy_file(
        "embeddings/mouse_embeddings.npy",
        "embeddings/mouse_iris_labels.npy"
    )

    combined_jaccard_matrix, combined_cosine_matrix, combined_iris, combined_labels = project_combined_class(
        human_iris, human_labels, human_embeddings, mouse_iris, mouse_labels, mouse_embeddings
    )

    assert combined_jaccard_matrix.shape == combined_cosine_matrix.shape, "Combined matrices must have the same shape."

    combined_scores = compute_average_similarities(combined_jaccard_matrix, combined_cosine_matrix)

    print("Combined Scores:\n")
    display_similarity_matrix(combined_scores, combined_labels)
    
def create_embeddings_files():
    create_embedding_file_from_owl_file(
        "anatomy-dataset/anatomy-dataset/human.owl",
        "embeddings/human_embeddings.npy",
        "embeddings/human_iris_labels.npy"
    )
    create_embedding_file_from_owl_file(
        "anatomy-dataset/anatomy-dataset/mouse.owl",
        "embeddings/mouse_embeddings.npy",
        "embeddings/mouse_iris_labels.npy"
    )


## Create the Embeddings
#create_embeddings_files()

## Run the Program
main()