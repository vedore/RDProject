import numpy as np
import torch
from py_files.cosine_methods import compare_embeddings_cosine_nearest_neighbor
from py_files.embeddings_extractor import create_embedding_file_from_owl_file, get_data_from_npy_file
from py_files.jaccard_methods import compare_labels_jaccard


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


def project_class(embeddings_file, iris_label_file):
    iris, labels, average_embeddings = get_data_from_npy_file(embeddings_file, iris_label_file)

    # Calculate Jaccard similarity scores for labels
    print("Jaccard Similarity Step")
    jaccard_similarity_matrix = compare_labels_jaccard(labels)

    # Calculate cosine similarity scores for embeddings
    print("Cosine Similarity Step")
    cosine_similarity_matrix = compare_embeddings_cosine_nearest_neighbor(average_embeddings)

    return jaccard_similarity_matrix, cosine_similarity_matrix, iris, labels

def main():
    print("Human Ontology Process:\n")
    human_jaccard_matrix, human_cosine_matrix, human_iris, human_labels = project_class(
        embeddings_file="embeddings/human_embeddings.npy",
        iris_label_file="embeddings/human_iris_labels.npy"
    )

    print("Mouse Ontology Process:\n")
    mouse_jaccard_matrix, mouse_cosine_matrix, mouse_iris, mouse_labels = project_class(
        embeddings_file="embeddings/mouse_embeddings.npy",
        iris_label_file="embeddings/mouse_iris_labels.npy"
    )

    # Ensure that both matrices have the same shape
    assert human_jaccard_matrix.shape == human_cosine_matrix.shape, "Human matrices must have the same shape."
    assert mouse_jaccard_matrix.shape == mouse_cosine_matrix.shape, "Mouse matrices must have the same shape."

    combined_scores_human = compute_average_similarities(human_jaccard_matrix, human_cosine_matrix)
    combined_scores_mouse = compute_average_similarities(mouse_jaccard_matrix, mouse_cosine_matrix)

    print("Human Combined Scores:\n", combined_scores_human)
    print("Mouse Combined Scores:\n", combined_scores_mouse)
    
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
create_embeddings_files()

## Run the Program
main()