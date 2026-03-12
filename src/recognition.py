"""
recognition.py — Nearest-neighbour classification in PCA space.

Linear-algebra context
----------------------
After projecting every face into the k-dimensional eigenface subspace
we obtain compact *weight vectors*  w ∈ ℝ^k .  Classification is
performed by finding the training face whose weight vector is closest
(in Euclidean distance) to the test face's weight vector:

    ŷ = y_{argmin_i ‖w_test − w_train_i‖₂}

This is a 1-Nearest-Neighbour (1-NN) classifier.  Despite its simplicity,
it performs well when PCA captures enough discriminative variance.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.eigenfaces import project


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_nearest_neighbor(X_train_proj, y_train, X_test_proj):
    """
    1-Nearest-Neighbour classification using Euclidean distance.

    Parameters
    ----------
    X_train_proj : ndarray, shape (n_train, k)
    y_train : ndarray, shape (n_train,)
    X_test_proj : ndarray, shape (n_test, k)

    Returns
    -------
    y_pred : ndarray, shape (n_test,)
        Predicted labels for the test set.
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X_train_proj, y_train)
    y_pred = knn.predict(X_test_proj)
    return y_pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    """
    Compute accuracy and per-class precision/recall/F1.

    Returns
    -------
    accuracy : float
    report : str
        Formatted classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return accuracy, report


# ---------------------------------------------------------------------------
# Accuracy sweep
# ---------------------------------------------------------------------------

def accuracy_vs_k(X_train_centered, y_train,
                  X_test_centered, y_test,
                  pca, component_counts):
    """
    Evaluate recognition accuracy across multiple component counts.

    For each k in *component_counts*, project train and test data into
    a k-dimensional PCA subspace and run 1-NN classification.

    Parameters
    ----------
    X_train_centered, X_test_centered : ndarray
        Centered face vectors.
    y_train, y_test : ndarray
        True labels.
    pca : fitted PCA object
    component_counts : list of int

    Returns
    -------
    ks : list of int
    accuracies : list of float
    best_k : int
        Component count that achieved the highest accuracy.
    best_acc : float
    """
    ks, accuracies = [], []
    for k in component_counts:
        if k > pca.n_components_:
            continue
        X_train_proj = project(X_train_centered, pca, n_components=k)
        X_test_proj  = project(X_test_centered,  pca, n_components=k)
        y_pred       = classify_nearest_neighbor(X_train_proj, y_train, X_test_proj)
        acc          = accuracy_score(y_test, y_pred)
        ks.append(k)
        accuracies.append(acc)

    best_idx = int(np.argmax(accuracies))
    best_k   = ks[best_idx]
    best_acc = accuracies[best_idx]
    return ks, accuracies, best_k, best_acc
