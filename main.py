#!/usr/bin/env python3
"""
Run this script to execute the full face-recognition workflow:

    python main.py

All figures are saved to  output/figures/  and a summary of results is printed to the console.
"""

import numpy as np

# ── Project modules ────────────────────────────────────────────────────────
from src.config import (
    COMPONENT_COUNTS, N_EIGENFACES_DISPLAY, RANDOM_SEED, ensure_output_dirs,
)
from src.data_loader      import load_olivetti_faces, split_data
from src.preprocessing     import compute_mean_face, center_data, vector_to_image
from src.eigenfaces        import compute_pca, get_eigenfaces, project
from src.reconstruction    import (
    reconstruct_faces, compute_mse, reconstruction_error_vs_k,
)
from src.recognition       import (
    classify_nearest_neighbor, evaluate, accuracy_vs_k,
)
from src import visualization as viz


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _section(title):
    """Print a nicely formatted section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ensure_output_dirs()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    _section("1 · Loading Dataset")
    images, X, y = load_olivetti_faces()
    print(f"  Dataset shape : {X.shape}  ({len(np.unique(y))} subjects)")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    _section("2 · Splitting Data")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    # ------------------------------------------------------------------
    # 3. Preprocessing — mean face & centering
    # ------------------------------------------------------------------
    _section("3 · Preprocessing")
    mean_face = compute_mean_face(X_train)
    X_train_c = center_data(X_train, mean_face)
    X_test_c  = center_data(X_test,  mean_face)
    print("  Computed mean face from training set.")
    print("  Centered train and test data.")

    # ------------------------------------------------------------------
    # 4. PCA / Eigenfaces
    # ------------------------------------------------------------------
    _section("4 · Computing PCA (Eigenfaces)")
    pca = compute_pca(X_train_c)
    eigenfaces = get_eigenfaces(pca)
    print(f"  Retained components : {pca.n_components_}")
    print(f"  Explained variance (first 5) : "
          f"{pca.explained_variance_ratio_[:5].round(4)}")
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    idx_90 = np.searchsorted(cum_var, 0.90) + 1
    idx_95 = np.searchsorted(cum_var, 0.95) + 1
    print(f"  Components for 90% variance  : {idx_90}")
    print(f"  Components for 95% variance  : {idx_95}")

    # ------------------------------------------------------------------
    # 5. Visualisations — dataset, mean face, eigenfaces, variance
    # ------------------------------------------------------------------
    _section("5 · Generating Visualisations")
    viz.plot_sample_faces(images, y)
    viz.plot_mean_face(mean_face)
    viz.plot_eigenfaces(eigenfaces, n=N_EIGENFACES_DISPLAY)
    viz.plot_explained_variance(pca)

    # Projections (2-D & 3-D)
    X_train_proj_full = project(X_train_c, pca)
    viz.plot_projection_2d(X_train_proj_full, y_train)
    viz.plot_projection_3d(X_train_proj_full, y_train)

    # ------------------------------------------------------------------
    # 6. Reconstruction
    # ------------------------------------------------------------------
    _section("6 · Face Reconstruction")
    recon_ks = [1, 5, 10, 20, 50, 100, 200, 300]
    recon_ks = [k for k in recon_ks if k <= pca.n_components_]

    # Build reconstructed images for a few sample faces
    sample_indices = [0, 1, 2, 3, 4]
    reconstructed_dict = {}
    for k in recon_ks:
        X_recon = reconstruct_faces(X_train_c, pca, k, mean_face)
        reconstructed_dict[k] = X_recon
        _, avg_mse = compute_mse(X_train, X_recon)
        print(f"  k = {k:>4d}  |  MSE = {avg_mse:.6f}")

    viz.plot_reconstruction_grid(X_train, reconstructed_dict, sample_indices)

    # Reconstruction error sweep (over all COMPONENT_COUNTS)
    ks_err, mses_err = reconstruction_error_vs_k(
        X_train_c, X_train, pca, mean_face, COMPONENT_COUNTS,
    )
    viz.plot_reconstruction_error(ks_err, mses_err)

    # ------------------------------------------------------------------
    # 7. Recognition / Classification
    # ------------------------------------------------------------------
    _section("7 · Face Recognition (1-NN)")
    ks_acc, accs, best_k, best_acc = accuracy_vs_k(
        X_train_c, y_train,
        X_test_c, y_test,
        pca, COMPONENT_COUNTS,
    )
    for k, acc in zip(ks_acc, accs):
        print(f"  k = {k:>4d}  |  Accuracy = {acc:.2%}")

    print()
    print(f"  ★ Best k = {best_k}  →  Accuracy = {best_acc:.2%}")

    viz.plot_accuracy_vs_k(ks_acc, accs, best_k, best_acc)

    # ------------------------------------------------------------------
    # 8. Error analysis — correctly & incorrectly classified
    # ------------------------------------------------------------------
    _section("8 · Error Analysis")
    # Classify at the best-k setting for error-analysis plots
    X_train_proj_best = project(X_train_c, pca, n_components=best_k)
    X_test_proj_best  = project(X_test_c,  pca, n_components=best_k)
    y_pred = classify_nearest_neighbor(X_train_proj_best, y_train, X_test_proj_best)
    accuracy, report = evaluate(y_test, y_pred)

    print(f"  Accuracy at best k ({best_k}): {accuracy:.2%}")
    print()
    print(report)

    viz.plot_correctly_classified(X_test, y_test, y_pred)
    viz.plot_misclassified(X_test, y_test, y_pred)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _section("Pipeline Complete")
    print("  All figures saved to output/figures/")
    print("  Done ✓")


if __name__ == "__main__":
    main()
