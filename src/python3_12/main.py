import datetime

from src.python3_12.methods.combine_similarities import combine_similarities
from src.python3_12.methods.embeddings_extractor import get_content_from_owl_file
from src.python3_12.methods.lexical_similarity import compute_lexical_similarity
from src.python3_12.methods.rdf_generator import generate_rdf_from_similarity
from src.python3_12.methods.semantic_similarity import compute_semantic_similarity


def compare_ontologies(first_owl_path, second_owl_path):

    start_time = datetime.datetime.now()

    first_list_of_classes = get_content_from_owl_file(owl_path=first_owl_path)
    second_list_of_classes = get_content_from_owl_file(owl_path=second_owl_path)

    first_labels_lower = [' '.join(str(label).lower() for label in labels) for iri, labels in first_list_of_classes]

    # print("Second Label")
    second_labels_lower = [' '.join(str(label).lower() for label in labels) for iri, labels in second_list_of_classes]

    lexical_similarity = compute_lexical_similarity(first_labels_lower, second_labels_lower)

    semantic_similarity = compute_semantic_similarity(first_labels_lower, second_labels_lower)

    # High alpha means that the combined similarity will rely more heavily on the lexical similarity.
    # This is useful when the textual content and term frequency are more important for your comparison.

    # Low alpha means that the combined similarity will rely more heavily on the semantic similarity.
    # For instance, if This is useful when the structure and presence of terms are more important.

    combined_similarity = combine_similarities(lexical_similarity, semantic_similarity, alpha=0.5)

    # rdf could have nothing if threshold to low
    rdf_triplet = generate_rdf_from_similarity(combined_similarity, first_labels_lower,
                                               second_labels_lower, 0.5, "rdf_file.rdf")

    print(rdf_triplet)

    # alignment_pairs = get_alignment_pairs(combined_similarity)

    # Print alignment pairs with scores
    # for iri1, labels1, iri2, labels2, score in alignment_pairs:
    #     print(f"Aligned: {iri1} -> {iri2} with score: {score}")

    end_time = datetime.datetime.now()

    print(end_time - start_time)


compare_ontologies("D:\\Faculdade\\RedesDeConhecimento\\RDProject\\ontologies\\human.owl",
                   "D:\\Faculdade\\RedesDeConhecimento\\RDProject\\ontologies\\mouse.owl")

