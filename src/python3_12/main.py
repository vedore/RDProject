import datetime

from src.python3_12.methods.embeddings_extractor import get_content_from_owl_file
from src.python3_12.methods.lexical_similarity import compute_lexical_similarity
from src.python3_12.methods.semantic_similarity import compute_semantic_similarity


def combine_similarities(lexical, semantic, alpha=0.5):
    combined_similarity = {}
    for (iri1, labels1, iri2, labels2, lex_sim) in lexical:
        combined_similarity[(iri1, labels1, iri2, labels2)] = alpha * lex_sim

    for (iri1, labels1, iri2, labels2, sem_sim) in semantic:
        if (iri1, labels1, iri2, labels2) in combined_similarity:
            combined_similarity[(iri1, labels1, iri2, labels2)] += (1 - alpha) * sem_sim
        else:
            combined_similarity[(iri1, labels1, iri2, labels2)] = (1 - alpha) * sem_sim

    return combined_similarity


def get_alignment_pairs(similarity_dict, threshold=0.7):
    alignment_pairs = [(iri1, labels1, iri2, labels2, score) for (iri1, labels1, iri2, labels2), score in similarity_dict.items() if score >= threshold]
    return alignment_pairs


def compare_ontologies(first_owl_path, second_owl_path):

    start_time = datetime.datetime.now()

    first_list_of_classes = get_content_from_owl_file(owl_path=first_owl_path)
    second_list_of_classes = get_content_from_owl_file(owl_path=second_owl_path)

    lexical_similarity = compute_lexical_similarity(first_list_of_classes, second_list_of_classes)

    # semantic_similarity = compute_semantic_similarity(first_list_of_classes, second_list_of_classes)

    # combined_similarity = combine_similarities(lexical_similarity, semantic_similarity, alpha=0.7)

    # alignment_pairs = get_alignment_pairs(combined_similarity)

    # Print alignment pairs with scores
    # for iri1, labels1, iri2, labels2, score in alignment_pairs:
    #     print(f"Aligned: {iri1} -> {iri2} with score: {score}")

    # end_time = datetime.datetime.now()

    # print(end_time - start_time)


compare_ontologies("D:\\Faculdade\\RedesDeConhecimento\\RDProject\\ontologies\\human.owl",
                   "D:\\Faculdade\\RedesDeConhecimento\\RDProject\\ontologies\\mouse.owl")
