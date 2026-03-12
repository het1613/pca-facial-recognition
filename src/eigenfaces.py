"""
eigenfaces.py — PCA computation, eigenface extraction, and projection.

Linear-algebra context
----------------------
Given a centered data matrix  X̃ ∈ ℝ^{N×d}  (N training faces, d = 4096
pixels), PCA computes the eigendecomposition of the covariance matrix

    C = (1/N) X̃ᵀ X̃   ∈ ℝ^{d×d}

The eigenvectors  v₁, v₂, …, vₖ  corresponding to the  k  largest
eigenvalues form an orthonormal basis for a  k-dimensional subspace that
captures the maximum variance of the data.  When reshaped back into
64 × 64 images, these eigenvectors are the **eigenfaces**.

In practice scikit-learn's PCA class uses the *Singular Value Decomposition*
(SVD) of X̃ rather than explicitly forming C, which is numerically more
stable:

    X̃ = U Σ Vᵀ

The rows of Vᵀ (i.e. the right-singular vectors) are exactly the
eigenvectors of C, and the squared singular values  σ²  are proportional to
the eigenvalues.

Projection & reconstruction
----------------------------
To project a face  x  into PCA space we compute its *weight vector*:

    w = Vₖᵀ (x − μ)     ∈ ℝ^k

where Vₖ contains only the top-k eigenvectors.  The reconstructed face is:

    x̂ = μ + Vₖ w
"""

import numpy as np
from sklearn.decomposition import PCA

from src.config import IMAGE_SHAPE


# ---------------------------------------------------------------------------
# PCA computation
# ---------------------------------------------------------------------------

def compute_pca(X_centered, n_components=None):
    """
    Fit PCA on the centered training data.

    Parameters
    ----------
    X_centered : ndarray, shape (n_train, n_features)
        Mean-subtracted training face vectors.
    n_components : int or None
        Number of principal components to retain.
        None → keep min(n_samples, n_features) components.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.  Key attributes:
        - ``components_``      : eigenvectors / eigenfaces  (k × d)
        - ``explained_variance_ratio_`` : fraction of total variance
          captured by each component.
        - ``mean_``  is *not* used because we center manually.
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

    Parameters
    ----------
    pca : fitted PCA object
    image_shape : tuple of int
        (height, width) of a single face image.

    Returns
    -------
    eigenfaces : ndarray, shape (n_components, H, W)
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

    If *n_components* is smaller than the number of components stored in
    ``pca``, we take only the first *n_components* eigenvectors to form the
    projection matrix  Vₖ , and compute  w = Vₖᵀ x̃  for each sample.

    Parameters
    ----------
    X_centered : ndarray, shape (n_samples, n_features)
    pca : fitted PCA object
    n_components : int or None
        Number of components to use for the projection.
        None → use all components available in *pca*.

    Returns
    -------
    X_proj : ndarray, shape (n_samples, n_components)
        Coordinates of each face in the reduced PCA subspace.
    """
    if n_components is None:
        n_components = pca.n_components_

    # Vₖ — the first k eigenvectors (rows of pca.components_)
    V_k = pca.components_[:n_components]   # shape (k, d)

    # w_i = Vₖ · x̃_i   for every sample → matrix multiply  X̃ Vₖᵀ
    X_proj = X_centered @ V_k.T            # shape (N, k)
    return X_proj
