import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from py_files.cosine_methods import compare_embeddings_cosine_nearest_neighbor
from py_files.embeddings_extractor import create_embedding_file_from_owl_file, get_data_from_npy_file
from py_files.jaccard_methods import compare_labels_jaccard
import pandas as pd
import pickle


def compute_average_similarities(jaccard_scores, cosine_scores):
    # Ensure the matrices are of the same size
    assert jaccard_scores.shape == cosine_scores.shape, "The similarity matrices must have the same shape."
    
    # Compute the average similarity scores
    avg_similarities = (jaccard_scores + cosine_scores) / 2
    
    return avg_similarities


def compute_average_lexical_similarities(jaccard_scores):
    return sum(jaccard_scores) / len(jaccard_scores)


def compute_average_embedding_similarities(cosine_scores):
    return sum(cosine_scores) / len(cosine_scores)


def project_combined_class(human_iris, human_labels, human_embeddings, mouse_iris, mouse_labels, mouse_embeddings):
    combined_iris = human_iris + mouse_iris
    combined_labels = human_labels + mouse_labels
    combined_embeddings = np.concatenate((human_embeddings, mouse_embeddings), axis=0)

    print("Combined Jaccard Similarity Step")
    combined_jaccard_similarity_matrix = compare_labels_jaccard(combined_labels)

    print("Combined Cosine Similarity Step")
    combined_cosine_similarity_matrix = compare_embeddings_cosine_nearest_neighbor(combined_embeddings)

    return combined_jaccard_similarity_matrix, combined_cosine_similarity_matrix, combined_iris, combined_labels


def display_similarity_matrix(similarity_matrix, labels):
    df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    print(df)


def prepare_data(human_embeddings, human_labels, mouse_embeddings, mouse_labels):
    # Combine embeddings and labels
    embeddings = np.concatenate((human_embeddings, mouse_embeddings), axis=0)
    labels = np.concatenate((human_labels, mouse_labels), axis=0)
    return train_test_split(embeddings, labels, test_size=0.2, random_state=42)


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=10,  min_samples_split=4, min_samples_leaf=2, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


def main():

    print("Human Ontology Process:\n")
    human_iris, human_labels, human_embeddings = get_data_from_npy_file(
        "embeddings/human_embeddings.npy",
        "embeddings/human_iris_labels.npy"
    )

    print("Mouse Ontology Process:\n")
    mouse_iris, mouse_labels, mouse_embeddings = get_data_from_npy_file(
        "embeddings/mouse_embeddings.npy",
        "embeddings/mouse_iris_labels.npy"
    )

    # combined_jaccard_matrix, combined_cosine_matrix, combined_iris, combined_labels = project_combined_class(
    #     human_iris, human_labels, human_embeddings, mouse_iris, mouse_labels, mouse_embeddings
    # )

    # assert combined_jaccard_matrix.shape == combined_cosine_matrix.shape, "Combined matrices must have the same shape."

    # combined_scores = compute_average_similarities(combined_jaccard_matrix, combined_cosine_matrix)

    # print("Combined Scores:\n")
    # display_similarity_matrix(combined_scores, combined_labels)


    print("Prepare The Data")
    X_train, X_test, y_train, y_test = prepare_data(human_embeddings, human_labels, mouse_embeddings, mouse_labels)

    print("Training")
    rf_model = train_random_forest(X_train, y_train)

    with open("random_forest_model\\rfmodel.pkl", 'wb') as f:
        pickle.dump(rf_model, f)

    print("Dumped The Model")

    print("Evaluating")
    precision, recall, f1 = evaluate_model(rf_model, X_test, y_test)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


    # Random Forest

    
def create_embeddings_files():
    create_embedding_file_from_owl_file(
        "anatomy-dataset/anatomy-dataset/human.owl",
        "embeddings/human_embeddings.npy",
        "embeddings/human_iris_labels.npy"
    )
    create_embedding_file_from_owl_file(
        "anatomy-dataset/anatomy-dataset/mouse.owl",
        "embeddings/mouse_embeddings.npy",
        "embeddings/mouse_iris_labels.npy"
    )


## Create the Embeddings
#create_embeddings_files()

## Run the Program
main()