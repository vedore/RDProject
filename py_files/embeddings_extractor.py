import os
import torch
import numpy as np

from owlready2 import *
from transformers import BertModel, BertTokenizer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def get_list_of_labels_from_owl_file(owl_path):
    onto = get_ontology("file://" + owl_path).load()
    list_of_labels = []

    for obj in onto.classes():
        if hasattr(obj, "label"):
            labels = obj.label
            for label in labels:
                list_of_labels.append(label)

    return list_of_labels


def get_model_for_embedding(model_name):
    return BertModel.from_pretrained(model_name)


def get_tokenizer_for_embedding(model_name):
    return BertTokenizer.from_pretrained(model_name)


def create_embedding_file_from_owl_file(owl_path, npy_file_name):
        
    start_time = datetime.now()

    print("Started To List The Labels")

    list_of_labels = get_list_of_labels_from_owl_file(owl_path)

    print("Started To Get The Model And Tokenizer")

    model_name = "bert-base-uncased"
    model = get_model_for_embedding(model_name)
    tokenizer = get_tokenizer_for_embedding(model_name)

    print("Started To Tokenize The Labels")

    tokenized_labels = [tokenizer(lbl, return_tensors="pt") for lbl in list_of_labels]

    print("Start The Embedding")

    with torch.no_grad():
        embeddings = [model(**lbl)["last_hidden_state"].squeeze(0) for lbl in tokenized_labels]

    averaged_embeddings = [torch.mean(embedding, dim=0) for embedding in embeddings]

    np.save(npy_file_name, averaged_embeddings)

    finish_time = datetime.now()

    print("Embedding Ended in: " + str((finish_time - start_time).total_seconds()))


def get_average_embeddings_from_npy_file(npy_path):
    return np.load(npy_path)


def get_embedding_from_owl_file(owl_path):
    start_time = datetime.now()

    print("Started To List The Labels")

    list_of_labels = get_list_of_labels_from_ontology(owl_path)

    print("Started To Get The Model And Tokenizer")

    model_name = "bert-base-uncased"
    model = get_model_for_embedding(model_name)
    tokenizer = get_tokenizer_for_embedding(model_name)

    print("Started To Tokenize The Labels")

    tokenized_labels = [tokenizer(lbl, return_tensors="pt") for lbl in list_of_labels]

    print("Start The Embedding")

    with torch.no_grad():
        embeddings = [model(**lbl)["last_hidden_state"].squeeze(0) for lbl in tokenized_labels]

    averaged_embeddings = [torch.mean(embedding, dim=0) for embedding in embeddings]

    finish_time = datetime.now()

    print("Embedding Ended in: " + str((finish_time - start_time).total_seconds()))

    return list_of_labels, averaged_embeddings


def get_embedding_file(owl_path, npy_file_name):
    labels = get_list_of_labels_from_owl_file(owl_path=owl_path)
    average_embeddings = get_average_embeddings_from_npy_file(npy_path=npy_file_name)
    
    # Calculate Jaccard similarity scores for labels
    jaccard_similarity_scores = compare_labels_jaccard(labels)
    
    # Calculate cosine similarity scores for embeddings
    cosine_similarity_scores = compare_embeddings_cosine_nearest_neighbor(average_embeddings)
    
    return labels, average_embeddings, jaccard_similarity_scores, cosine_similarity_scores


## Adicionado agora
def compute_average_similarities(jaccard_scores, cosine_scores):
    # Ensure the matrices are of the same size
    assert jaccard_scores.shape == cosine_scores.shape, "The similarity matrices must have the same shape."
    
    # Compute the average similarity scores
    avg_similarities = (jaccard_scores + cosine_scores) / 2
    
    return avg_similarities



# Example usage
# human_labels, human_embeddings, human_jaccard_similarity_scores, human_cosine_similarity_scores = get_embedding_file("anatomy-dataset/anatomy-dataset/human.owl", "human_embeddings.npy")
# mouse_labels, mouse_embeddings, mouse_jaccard_similarity_scores, mouse_cosine_similarity_scores = get_embedding_file("anatomy-dataset/anatomy-dataset/mouse.owl", "mouse_embeddings.npy")

# Now you have all the required data and similarity scores for both human and mouse datasets
# print("Human labels:", human_labels)
# print("Mouse labels:", mouse_labels)
# print("Human embeddings:", human_embeddings)
# print("Mouse embeddings:", mouse_embeddings)
# print("Human Jaccard similarity scores:", human_jaccard_similarity_scores)
# print("Mouse Jaccard similarity scores:", mouse_jaccard_similarity_scores)
# print("Human cosine similarity scores:", human_cosine_similarity_scores)
# print("Mouse cosine similarity scores:", mouse_cosine_similarity_scores)

# average_similarities = compute_average_similarities(human_jaccard_similarity_scores, human_cosine_similarity_scores)

# Display results
# print("Average Similarities:")
# for i, label1 in enumerate(human_labels):
#     for j, label2 in enumerate(human_labels):
#        print(f"{label1} - {label2}: {average_similarities[i, j]:.4f}")

