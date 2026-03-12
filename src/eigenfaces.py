import numpy as np
from sklearn.decomposition import PCA

from src.config import IMAGE_SHAPE


# ---------------------------------------------------------------------------
# PCA computation
# ---------------------------------------------------------------------------

def compute_pca(X_centered, n_components=None):
    """
    Fit PCA on the centered training data.
    """
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(X_centered)
    return pca


# ---------------------------------------------------------------------------
# Eigenface extraction
# ---------------------------------------------------------------------------

def get_eigenfaces(pca, image_shape=IMAGE_SHAPE):
    """
    Reshape the principal components into 2-D face images (eigenfaces).
    """
    # pca.components_ has shape (k, d).  Each row is an eigenvector.
    eigenfaces = pca.components_.reshape(-1, *image_shape)
    return eigenfaces


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project(X_centered, pca, n_components=None):
    """
    Project centered face vectors into PCA space.
    """
    if n_components is None:
        n_components = pca.n_components_

    # Vₖ — the first k eigenvectors (rows of pca.components_)
    V_k = pca.components_[:n_components]   # shape (k, d)

    # w_i = Vₖ · x̃_i   for every sample → matrix multiply  X̃ Vₖᵀ
    X_proj = X_centered @ V_k.T            # shape (N, k)
    return X_proj
