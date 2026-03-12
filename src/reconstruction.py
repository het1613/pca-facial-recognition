import numpy as np
from src.eigenfaces import project


def reconstruct_faces(X_centered, pca, n_components, mean_face):
    """
    Reconstruct faces using the first *n_components* principal components.
    """
    V_k = pca.components_[:n_components]   # (k, d)

    # Project into PCA space and back:  x̂_centered = Vₖ (Vₖᵀ x̃)
    X_proj   = X_centered @ V_k.T          # (N, k)
    X_recon  = X_proj @ V_k                # (N, d) — back to pixel space (centered)

    # Add the mean face back to get reconstructed face in original scale
    X_reconstructed = X_recon + mean_face
    return X_reconstructed


def compute_mse(original, reconstructed):
    """
    Compute per-sample and average Mean Squared Error.
    """
    diff = original - reconstructed
    per_sample_mse = np.mean(diff ** 2, axis=1)
    avg_mse = np.mean(per_sample_mse)
    return per_sample_mse, avg_mse


def reconstruction_error_vs_k(X_centered, X_original, pca, mean_face,
                                component_counts):
    """
    Sweep over multiple k values and compute average reconstruction MSE for each.
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
