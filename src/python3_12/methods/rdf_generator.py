from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS


def generate_rdf_from_similarity(combined_similarity_matrix, labels1, labels2, threshold, output_file, rdf_format='xml'):
    similar_pairs = []

    # Iterate through the combined similarity matrix
    for i in range(combined_similarity_matrix.shape[0]):
        for j in range(combined_similarity_matrix.shape[1]):
            similarity_score = combined_similarity_matrix[i, j]

            # Check if the similarity score exceeds the threshold
            if similarity_score >= threshold:
                similar_pairs.append((labels1[i], labels2[j]))

    # Create an RDF graph
    graph = Graph()

    # Define namespaces
    ns = Namespace("http://RedesDeConhimento.org/ontology#")

    # Add triples to the graph
    for label1, label2 in similar_pairs:
        # Create URIRefs for subjects and predicates
        uri_part, additional_info = label1.split(' ')
        subject = URIRef(ns[uri_part])
        predicate = URIRef(RDF.type)  # You can use appropriate predicates here
        obj = Literal(label2)

        # Add the triple to the graph
        graph.add((subject, predicate, obj))

    # Serialize the graph to RDF file
    graph.serialize(destination=output_file, format=rdf_format)

