import os
import torch

from owlready2 import *
from transformers import BertModel, BertTokenizer
from datetime import datetime


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


def get_embedding_for_mouse_onto():
    start_time = datetime.now()

    mouse_owl_file = "D:\\Faculdade\\RedesDeConhecimento\\RDProject\\anatomy-dataset\\anatomy-dataset\\mouse.owl"

    # print("Started To List The Labels")

    list_of_labels = get_list_of_labels_from_ontology(mouse_owl_file)

    # print("Started To Get The Model And Tokenizer")

    model_name = "bert-base-uncased"
    model = get_model_for_embedding(model_name)
    tokenizer = get_tokenizer_for_embedding(model_name)

    # print("Started To Tokenize The Labels")

    tokenized_labels = [tokenizer(lbl, return_tensors="pt") for lbl in list_of_labels]

    # print("Start The Embedding")

    with torch.no_grad():
        embeddings = [model(**lbl)["last_hidden_state"].squeeze(0) for lbl in tokenized_labels]

    averaged_embeddings = [torch.mean(embedding, dim=0) for embedding in embeddings]

    finish_time = datetime.now()

    print("Embedding For Mouse Ended in: " + str((finish_time - start_time).total_seconds()))


def get_embedding_for_human_onto():
    start_time = datetime.now()

    mouse_owl_file = "D:\\Faculdade\\RedesDeConhecimento\\RDProject\\anatomy-dataset\\anatomy-dataset\\human.owl"

    # print("Started To List The Labels")

    list_of_labels = get_list_of_labels_from_ontology(mouse_owl_file)

    # print("Started To Get The Model And Tokenizer")

    model_name = "bert-base-uncased"
    model = get_model_for_embedding(model_name)
    tokenizer = get_tokenizer_for_embedding(model_name)

    # print("Started To Tokenize The Labels")

    tokenized_labels = [tokenizer(label, return_tensors="pt") for label in list_of_labels]

    # print("Start The Embedding")

    with torch.no_grad():
        embeddings = [model(**label)["last_hidden_state"].squeeze(0) for label in tokenized_labels]

    averaged_embeddings = [torch.mean(embedding, dim=0) for embedding in embeddings]

    finish_time = datetime.now()

    print("Embedding For Human Ended in: " + str((finish_time - start_time).total_seconds()))


get_embedding_for_human_onto()
get_embedding_for_mouse_onto()
