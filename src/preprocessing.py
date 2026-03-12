"""
preprocessing.py — Mean-face computation, data centering, and reshaping helpers.

Linear-algebra context
----------------------
In the Eigenfaces framework every face image is treated as a high-dimensional
vector (ℝ^d where d = 64 × 64 = 4096).  Before performing PCA we *center* the
data by subtracting the **mean face**:

    x̃_i = x_i − μ          where μ = (1/N) Σ x_i

This centering step is critical because PCA finds directions of maximum
*variance*, and centering ensures the covariance matrix

    C = (1/N) X̃ᵀ X̃

is computed correctly.  The eigenvectors of C (≡ the principal components)
are exactly the **eigenfaces**.
"""

import numpy as np

from src.config import IMAGE_SHAPE


# ---------------------------------------------------------------------------
# Mean face
# ---------------------------------------------------------------------------

def compute_mean_face(X_train):
    """
    Compute the mean face from the training set.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Flattened training face vectors.

    Returns
    -------
    mean_face : ndarray, shape (n_features,)
        Element-wise average across all training faces.
    """
    # μ = (1/N) Σ x_i — a single vector in ℝ^d
    mean_face = np.mean(X_train, axis=0)
    return mean_face


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------

def center_data(X, mean_face):
    """
    Center the data by subtracting the mean face.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    mean_face : ndarray, shape (n_features,)

    Returns
    -------
    X_centered : ndarray, shape (n_samples, n_features)
        Zero-mean version of X.
    """
    # Broadcasting: each row x_i gets μ subtracted element-wise.
    X_centered = X - mean_face
    return X_centered


# ---------------------------------------------------------------------------
# Reshaping helpers
# ---------------------------------------------------------------------------

def vector_to_image(v, shape=IMAGE_SHAPE):
    """
    Reshape a flattened vector back into a 2-D image.

    Parameters
    ----------
    v : ndarray, shape (n_pixels,)
    shape : tuple of int
        Target (height, width).  Default is (64, 64).

    Returns
    -------
    img : ndarray, shape ``shape``
    """
    return v.reshape(shape)


def image_to_vector(img):
    """
    Flatten a 2-D image into a 1-D vector.

    Parameters
    ----------
    img : ndarray, shape (H, W)

    Returns
    -------
    v : ndarray, shape (H*W,)
    """
    return img.flatten()
