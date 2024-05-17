import numpy as np
import torch
from py_files.cosine_methods import compare_embeddings_cosine_nearest_neighbor
from py_files.embeddings_extractor import get_list_of_labels_from_owl_file, get_model_for_embedding, get_tokenizer_for_embedding
from py_files.jaccard_methods import compare_labels_jaccard


def compute_average_similarities(jaccard_scores, cosine_scores):

    ## IMPOSSIVEL DE FAZER POR CAUSA DOS DIFERENTES SHAPES

    # Convert lists to NumPy arrays
    jaccard_scores = np.array(jaccard_scores)
    cosine_scores = np.array(cosine_scores)

    print(jaccard_scores.shape)
    print(cosine_scores.shape)

    # Ensure the matrices are of the same size
    assert jaccard_scores.shape == cosine_scores.shape, "The similarity matrices must have the same shape."
    
    # Compute the average similarity scores
    avg_similarities = (jaccard_scores + cosine_scores) / 2
    
    return avg_similarities

def compute_average_lexical_similarities(jaccard_scores):
    return sum(jaccard_scores) / len(jaccard_scores)

def compute_average_embedding_similarities(cosine_scores):
    return sum(cosine_scores) / len(cosine_scores)


def project_class(owl_path, npy_file):

    ont_location = owl_path
    embedding_location = npy_file

    print("Labels Step")
    labels = get_list_of_labels_from_owl_file(ont_location)

    print("Average Embeddings Step")
    average_embeddings = np.load(embedding_location)

    # Calculate Jaccard similarity scores for labels
    print("Jaccard Similarity Step")
    jaccard_similarity_scores = compare_labels_jaccard(labels)

    # Calculate cosine similarity scores for embeddings
    print("Cosine Similarity Step")
    cosine_similarity_scores = compare_embeddings_cosine_nearest_neighbor(average_embeddings)

    print("Average Similarities Step")
    average_jaccard_score = compute_average_lexical_similarities(jaccard_similarity_scores)
    average_cosine_score = compute_average_embedding_similarities(cosine_similarity_scores)

    # Now you have all the required data and similarity scores for both human and mouse datasets
    print("Labels:", labels)
    print("Average Embeddings:", average_embeddings)
    print("Jaccard Similarity Scores:", jaccard_similarity_scores)
    print("Cosine Similarity Scores:", cosine_similarity_scores)
    print("Average Jaccard:", average_jaccard_score)
    print("Average Cosine:", average_cosine_score)


def main():

    print("Human Ontology Process:\n")
    project_class(owl_path="anatomy-dataset\\anatomy-dataset\\human.owl", npy_file="embeddings\\human_embeddings.npy")

    print("Mouse Ontology Process:\n")
    project_class(owl_path="anatomy-dataset\\anatomy-dataset\\mouse.owl", npy_file="embeddings\\mouse_embeddings.npy")
    


main()