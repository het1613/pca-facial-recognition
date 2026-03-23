import numpy as np

from src.config import IMAGE_SHAPE


def compute_mean_face(X_train):
    """
    Compute the mean face from the training set.
    """
    # μ = (1/N) Σ x_i - a single vector in ℝ^d
    mean_face = np.mean(X_train, axis=0)
    return mean_face


def center_data(X, mean_face):
    """
    Center the data by subtracting the mean face.
    """
    # Broadcasting: each row x_i gets μ subtracted element-wise.
    X_centered = X - mean_face
    return X_centered


def vector_to_image(v, shape=IMAGE_SHAPE):
    """
    Reshape a flattened vector back into a 2-D image.
    """
    return v.reshape(shape)


def image_to_vector(img):
    """
    Flatten a 2-D image into a 1-D vector.
    """
    return img.flatten()
