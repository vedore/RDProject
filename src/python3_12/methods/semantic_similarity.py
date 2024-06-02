from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_sentence_embedding(sentence):
    return model.encode(sentence, convert_to_tensor=True)


def compute_semantic_similarity(classes1, classes2):
    similarity_matrix = []
    for iri1, labels1 in classes1:
        embedding1 = get_sentence_embedding(labels1)
        for iri2, labels2 in classes2:
            embedding2 = get_sentence_embedding(labels2)
            similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
            similarity_matrix.append((iri1, labels1, iri2, labels2, similarity))
    return similarity_matrix
