from owlready2 import *

from src.python3_12.methods.model_methods import get_model_for_embedding, get_tokenizer_for_embedding


def get_content_from_owl_file(owl_path):

    onto = get_ontology("file://" + owl_path).load()

    list_of_classes = []

    for cls in onto.classes():

        list_of_labels = []
        list_of_synonyms = []

        # Get the IRI of the class
        iri = cls.iri

        # Check for labels and append them to the list
        if hasattr(cls, "label"):
            for lbl in cls.label:
                list_of_labels.append(lbl)

        # Check for the synonyms and append them to the list
        if hasattr(cls, "hasRelatedSynonym"):
            for synonym in cls.hasRelatedSynonym:
                list_of_synonyms.append(synonym)

        list_of_classes.append((iri, list_of_labels + list_of_synonyms))

    # print(list_of_classes)

    return list_of_classes

