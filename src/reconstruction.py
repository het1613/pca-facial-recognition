"""
reconstruction.py — Face reconstruction from PCA components and error analysis.

Linear-algebra context
----------------------
Given the projection  w = Vₖᵀ (x − μ)  and the top-k eigenvectors Vₖ ,
the reconstructed face vector is:

    x̂ = μ  +  Vₖ w
      = μ  +  Vₖ Vₖᵀ (x − μ)

This is the *orthogonal projection* of the centered face onto the
k-dimensional eigenface subspace, shifted back by the mean face.

As k increases, x̂ → x  and the reconstruction error (MSE) decreases
monotonically.  With k = rank(X̃) the reconstruction is exact
(zero error).
"""

import numpy as np

from src.eigenfaces import project


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_faces(X_centered, pca, n_components, mean_face):
    """
    Reconstruct faces using the first *n_components* principal components.

    x̂_i = μ + Vₖ Vₖᵀ x̃_i

    Parameters
    ----------
    X_centered : ndarray, shape (n_samples, n_features)
    pca : fitted PCA object
    n_components : int
    mean_face : ndarray, shape (n_features,)

    Returns
    -------
    X_reconstructed : ndarray, shape (n_samples, n_features)
        Reconstructed face vectors (in original pixel space, not centered).
    """
    V_k = pca.components_[:n_components]   # (k, d)

    # Project into PCA space and back:  x̂_centered = Vₖ (Vₖᵀ x̃)
    X_proj   = X_centered @ V_k.T          # (N, k)
    X_recon  = X_proj @ V_k                # (N, d) — back to pixel space (centered)

    # Add the mean face back to get reconstructed face in original scale
    X_reconstructed = X_recon + mean_face
    return X_reconstructed


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------

def compute_mse(original, reconstructed):
    """
    Compute per-sample and average Mean Squared Error.

    MSE_i = (1/d) ‖x_i − x̂_i‖²

    Parameters
    ----------
    original : ndarray, shape (n_samples, n_features)
    reconstructed : ndarray, shape (n_samples, n_features)

    Returns
    -------
    per_sample_mse : ndarray, shape (n_samples,)
    avg_mse : float
    """
    diff = original - reconstructed
    per_sample_mse = np.mean(diff ** 2, axis=1)
    avg_mse = np.mean(per_sample_mse)
    return per_sample_mse, avg_mse


def reconstruction_error_vs_k(X_centered, X_original, pca, mean_face,
                                component_counts):
    """
    Sweep over multiple k values and compute average reconstruction MSE
    for each.

    Parameters
    ----------
    X_centered : ndarray, shape (n_samples, n_features)
    X_original : ndarray, shape (n_samples, n_features)
        Original (non-centered) face vectors — used as the ground truth.
    pca : fitted PCA object
    mean_face : ndarray, shape (n_features,)
    component_counts : list of int
        Values of k to evaluate.

    Returns
    -------
    ks : list of int
        Filtered component counts (capped at pca.n_components_).
    mses : list of float
        Average MSE for each k.
    """
    ks, mses = [], []
    for k in component_counts:
        if k > pca.n_components_:
            continue
        X_recon = reconstruct_faces(X_centered, pca, k, mean_face)
        _, avg_mse = compute_mse(X_original, X_recon)
        ks.append(k)
        mses.append(avg_mse)
    return ks, mses
