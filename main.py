import numpy as np
import torch
from py_files.cosine_methods import compare_embeddings_cosine_nearest_neighbor
from py_files.embeddings_extractor import create_embedding_file_from_owl_file, get_data_from_npy_file
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


def project_class(npy_file):

    iris, labels, average_embeddings = get_data_from_npy_file(npy_file)

    # Calculate Jaccard similarity scores for labels
    print("Jaccard Similarity Step")
    jaccard_similarity_scores = compare_labels_jaccard(labels)

    # Calculate cosine similarity scores for embeddings
    print("Cosine Similarity Step")
    cosine_similarity_scores = compare_embeddings_cosine_nearest_neighbor(average_embeddings)

    # Combine data into a structured format
    results = []

    for i, (iri, label) in enumerate(zip(iris, labels)):
        results.append({
            "IRI:source": iri,
            "label:source": label,
            "cosine_score": cosine_similarity_scores[i],
            "jaccard_score": jaccard_similarity_scores[i]
        })

    # Save or process the results as needed
    print("Results:", results)

    return results


def main():

    print("Human Ontology Process:\n")
    human_results = project_class(npy_file="embeddings\\human_embeddings.npy")

    print("Mouse Ontology Process:\n")
    mouse_results = project_class(npy_file="embeddings\\mouse_embeddings.npy")

    combined_scores = compute_average_similarities(human_results, mouse_results)
    print(combined_scores)
    
def create_embeddings_files():
    create_embedding_file_from_owl_file("anatomy-dataset\\anatomy-dataset\\human.owl", "embeddings\\human_embeddings.npy")
    create_embedding_file_from_owl_file("anatomy-dataset\\anatomy-dataset\\mouse.owl", "embeddings\\mouse_embeddings.npy")


## Create the Embeddings
## create_embeddings_files()

## Run the Program
main()