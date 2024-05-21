import os
import torch
import numpy as np

from owlready2 import *
from transformers import BertModel, BertTokenizer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def get_list_of_iris_and_labels_from_owl_file(owl_path):
    onto = get_ontology("file://" + owl_path).load()
    list_of_iris_labels = []


    for obj in onto.classes():

        if hasattr(obj, "label"):
            labels = obj.label
            for label in labels:
                list_of_iris_labels.append((obj.iri,label))

    return list_of_iris_labels


def get_model_for_embedding(model_name):
    return BertModel.from_pretrained(model_name)


def get_tokenizer_for_embedding(model_name):
    return BertTokenizer.from_pretrained(model_name)


def create_embedding_file_from_owl_file(owl_path, npy_file_name):
        
    start_time = datetime.now()

    print("Started To List The Labels")

    list_of_iris_labels = get_list_of_iris_and_labels_from_owl_file(owl_path)

    print("Started To Get The Model And Tokenizer")

    model_name = "bert-base-uncased"
    model = get_model_for_embedding(model_name)
    tokenizer = get_tokenizer_for_embedding(model_name)

    print("Started To Tokenize The Labels")

    tokenized_labels = [tokenizer(lbl[1], return_tensors="pt") for lbl in list_of_iris_labels]

    print("Start The Embedding")

    with torch.no_grad():
        embeddings = [model(**lbl)["last_hidden_state"].squeeze(0) for lbl in tokenized_labels]

    averaged_embeddings = [torch.mean(embedding, dim=0) for embedding in embeddings]

    # Save IRIs, labels, and embeddings
    data = [{"iris": lbl[0], "labels": lbl[1], "embeddings": averaged_embeddings} for lbl in list_of_iris_labels]
    np.save(npy_file_name, data)

    finish_time = datetime.now()

    print("Embedding Ended in: " + str((finish_time - start_time).total_seconds()))


def get_data_from_npy_file(npy_path):

    list_of_iris = []
    list_of_labels = []
    list_of_embeddings = []
    list_of_everything = []

    data = np.load(npy_path, allow_pickle=True)

    for entry in data:

        list_of_iris.append(entry['iris'])
        list_of_labels.append(entry['labels'])
        list_of_embeddings.append(entry['embeddings'])
        # list_of_everything.append(entry)

    return list_of_iris, list_of_labels, list_of_embeddings #, list_of_everything

    

