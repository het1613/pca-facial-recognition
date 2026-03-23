import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_SIZE


def load_olivetti_faces():
    """
    Load the Olivetti Faces dataset.
    """
    data = fetch_olivetti_faces(shuffle=False)
    images = data.images       # (400, 64, 64)
    X      = data.data         # (400, 4096)  - already flattened
    y      = data.target       # (400,)
    return images, X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Perform a stratified train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,                 # keep class proportions balanced
    )
    return X_train, X_test, y_train, y_test
