import os
import torch
import numpy as np

from owlready2 import *
from transformers import BertModel, BertTokenizer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_list_of_labels_from_ontology(owl_path):
    onto = get_ontology("file://" + owl_path).load()
    list_of_labels = []

    # Has alternate labels ?
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

def calculate_jaccard_similarity(label1, label2):
    # Tokenize labels into sets of words
    set1 = set(label1.lower().split())
    set2 = set(label2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_similarity = intersection / union
    return jaccard_similarity


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
    labels, embeddings = get_embedding_from_owl_file(owl_path)
    np.save(npy_file_name, embeddings)
    
    # Calculate Jaccard similarity scores for labels
    jaccard_similarity_scores = compare_labels_jaccard(labels)
    
    # Calculate cosine similarity scores for embeddings
    cosine_similarity_scores = compare_embeddings_cosine_nearest_neighbor(embeddings)
    
    return labels, embeddings, jaccard_similarity_scores, cosine_similarity_scores


def compare_labels_jaccard(labels):
    similarity_scores = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            similarity = calculate_jaccard_similarity(labels[i], labels[j])
            similarity_scores.append(similarity)

    return similarity_scores

def compare_embeddings_cosine_nearest_neighbor(embeddings):
    similarity_scores = []

    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(embeddings)

    for i in range(len(embeddings)):
        # Find nearest neighbors for each embedding
        distances, indices = nbrs.kneighbors([embeddings[i]])
        
        # Compute average cosine similarity with nearest neighbors
        avg_similarity = np.mean(1 - distances)
        similarity_scores.append(avg_similarity)

    return similarity_scores


# Example usage
human_labels, human_embeddings, human_jaccard_similarity_scores, human_cosine_similarity_scores = get_embedding_file("anatomy-dataset/anatomy-dataset/human.owl", "human_embeddings.npy")
mouse_labels, mouse_embeddings, mouse_jaccard_similarity_scores, mouse_cosine_similarity_scores = get_embedding_file("anatomy-dataset/anatomy-dataset/mouse.owl", "mouse_embeddings.npy")

# Now you have all the required data and similarity scores for both human and mouse datasets
print("Human labels:", human_labels)
print("Mouse labels:", mouse_labels)
print("Human embeddings:", human_embeddings)
print("Mouse embeddings:", mouse_embeddings)
print("Human Jaccard similarity scores:", human_jaccard_similarity_scores)
print("Mouse Jaccard similarity scores:", mouse_jaccard_similarity_scores)
print("Human cosine similarity scores:", human_cosine_similarity_scores)
print("Mouse cosine similarity scores:", mouse_cosine_similarity_scores)

