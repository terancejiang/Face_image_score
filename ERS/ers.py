"""""""""""""""""""""""""""""
Project: Face_image_score
Author: Terance Jiang
Date: 1/16/2024
"""""""""""""""""""""""""""""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calculate_ers(face_embedding, ui_centroid):
    """
    Calculate the Embedding Recognizability Score (ERS).

    Parameters:
    face_embedding: The embedding vector of the face image.
    ui_centroid: The average embedding vector of unrecognizable identity images.

    Returns:
    ERS as a float.
    """
    return 1 - euclidean_distances([face_embedding], [ui_centroid])[0][0]


def calculate_ui_centroid(ui_embeddings):
    """
    Calculate the centroid of UI embeddings.

    Parameters:
    ui_embeddings: A list of embedding vectors for UI images.

    Returns:
    The centroid of UI embeddings as a numpy array.
    """
    return np.mean(ui_embeddings, axis=0)


# Example usage:
# Assume 'face_embedding' is the embedding of a face image
# and 'ui_centroid' is the precomputed centroid of UI embeddings
ers_score = calculate_ers(face_embedding, ui_centroid)
