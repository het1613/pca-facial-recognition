import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.eigenfaces import project


def classify_nearest_neighbor(X_train_proj, y_train, X_test_proj):
    """
    1-Nearest-Neighbour classification using Euclidean distance.
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X_train_proj, y_train)
    y_pred = knn.predict(X_test_proj)
    return y_pred


def evaluate(y_true, y_pred):
    """
    Compute accuracy and per-class precision/recall/F1.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return accuracy, report


def accuracy_vs_k(X_train_centered, y_train,
                  X_test_centered, y_test,
                  pca, component_counts):
    """
    Evaluate recognition accuracy across multiple component counts.
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
