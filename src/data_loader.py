"""
data_loader.py — Dataset loading and train/test splitting.

Provides a clean interface to the Olivetti Faces dataset so that the rest of
the pipeline never interacts with scikit-learn's data-fetching API directly.
This makes it straightforward to swap in a different dataset later.
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_SIZE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_olivetti_faces():
    """
    Load the Olivetti Faces dataset.

    Returns
    -------
    images : ndarray, shape (400, 64, 64)
        Original 2-D face images (pixel values in [0, 1]).
    X : ndarray, shape (400, 4096)
        Flattened image vectors — each row is one face.
    y : ndarray, shape (400,)
        Integer subject labels (0–39, ten images per subject).
    """
    data = fetch_olivetti_faces(shuffle=False)
    images = data.images       # (400, 64, 64)
    X      = data.data         # (400, 4096)  — already flattened
    y      = data.target       # (400,)
    return images, X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Perform a stratified train/test split.

    Stratification ensures that every subject has roughly equal representation
    in both the training and test sets — important because the Olivetti dataset
    has only ten images per subject.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    test_size : float
        Fraction of data reserved for testing (default from config).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,                 # keep class proportions balanced
    )
    return X_train, X_test, y_train, y_test
