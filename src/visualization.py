import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 (needed for 3-D projection)

from src.config import (
    IMAGE_SHAPE, FIGURES_DIR, FIGURE_DPI, CMAP,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK,
    PROJECTION_PALETTE,
)

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        FONT_SIZE_TICK,
    "axes.titlesize":   FONT_SIZE_TITLE,
    "axes.labelsize":   FONT_SIZE_LABEL,
    "xtick.labelsize":  FONT_SIZE_TICK,
    "ytick.labelsize":  FONT_SIZE_TICK,
    "figure.dpi":       FIGURE_DPI,
    "savefig.dpi":      FIGURE_DPI,
    "savefig.bbox":     "tight",
    "figure.facecolor": "white",
})


def _save(fig, filename):
    """Save figure and close."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Sample faces overview
# ═══════════════════════════════════════════════════════════════════════════

def plot_sample_faces(images, labels, n_samples=20, shape=IMAGE_SHAPE):
    """
    Display a grid of sample face images with their subject labels.
    """
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.2 * n_rows))
    fig.suptitle("Sample Faces from the Olivetti Dataset", fontsize=FONT_SIZE_TITLE + 2, y=1.02)
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            img = images[i].reshape(shape) if images[i].ndim == 1 else images[i]
            ax.imshow(img, cmap=CMAP)
            ax.set_title(f"Subject {labels[i]}", fontsize=FONT_SIZE_TICK)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "01_sample_faces.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mean face
# ═══════════════════════════════════════════════════════════════════════════

def plot_mean_face(mean_face, shape=IMAGE_SHAPE):
    """
    Display the mean (average) face computed from the training set.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mean_face.reshape(shape), cmap=CMAP)
    ax.set_title("Mean Face (Training Set)")
    ax.axis("off")
    _save(fig, "02_mean_face.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Top eigenfaces
# ═══════════════════════════════════════════════════════════════════════════

def plot_eigenfaces(eigenfaces, n=16):
    """
    Display a grid of the top-n eigenfaces (principal components reshaped
    as images).
    """
    n = min(n, len(eigenfaces))
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows))
    fig.suptitle(f"Top {n} Eigenfaces (Principal Components)", fontsize=FONT_SIZE_TITLE + 2, y=1.02)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(eigenfaces[i], cmap=CMAP)
            ax.set_title(f"PC {i + 1}", fontsize=FONT_SIZE_TICK)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "03_top_eigenfaces.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Explained variance
# ═══════════════════════════════════════════════════════════════════════════

def plot_explained_variance(pca, max_components=100):
    """
    Plot individual and cumulative explained variance ratio.
    """
    n = min(max_components, len(pca.explained_variance_ratio_))
    individual = pca.explained_variance_ratio_[:n]
    cumulative = np.cumsum(pca.explained_variance_ratio_)[:n]
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_bar = "#4C72B0"
    color_line = "#DD8452"

    ax1.bar(x, individual, alpha=0.65, color=color_bar, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color=color_bar)
    ax1.tick_params(axis="y", labelcolor=color_bar)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color=color_line, linewidth=2.2, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance", color=color_line)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 1.05)

    # markers at 90 % and 95 %
    for threshold, ls in [(0.90, "--"), (0.95, ":")]:
        idx = np.searchsorted(cumulative, threshold)
        if idx < n:
            ax2.axhline(threshold, color="gray", linestyle=ls, linewidth=0.8)
            ax2.axvline(x[idx], color="gray", linestyle=ls, linewidth=0.8)
            ax2.annotate(f"{threshold*100:.0f}% at k={x[idx]}",
                         xy=(x[idx], threshold), fontsize=9,
                         xytext=(x[idx] + 3, threshold - 0.04),
                         arrowprops=dict(arrowstyle="->", color="gray"),
                         color="gray")

    fig.suptitle("Explained Variance by Principal Component", fontsize=FONT_SIZE_TITLE + 1)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94), fontsize=FONT_SIZE_TICK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "04_explained_variance.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. 2-D projection scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_projection_2d(X_proj, y):
    """
    Scatter plot of face projections onto the first two principal components.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(y)
    cmap = cm.get_cmap(PROJECTION_PALETTE, len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                   c=[cmap(idx)], label=str(label), s=30, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2-D PCA Projection of Faces")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2, title="Subject")
    fig.tight_layout()
    _save(fig, "05_projection_2d.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. 3-D projection scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_projection_3d(X_proj, y):
    """
    3-D scatter plot of face projections onto the first three PCs.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(y)
    cmap = cm.get_cmap(PROJECTION_PALETTE, len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1], X_proj[mask, 2],
                   c=[cmap(idx)], label=str(label), s=25, alpha=0.7, edgecolors="k", linewidths=0.2)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("3-D PCA Projection of Faces")
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=6, ncol=2, title="Subject")
    fig.tight_layout()
    _save(fig, "06_projection_3d.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Reconstruction grid
# ═══════════════════════════════════════════════════════════════════════════

def plot_reconstruction_grid(X_original, reconstructed_dict, indices,
                             shape=IMAGE_SHAPE):
    """
    Side-by-side comparison: original faces vs reconstructions at various k.

    Parameters
    ----------
    X_original : ndarray, shape (n_samples, n_features)
    reconstructed_dict : dict {k: ndarray}
        Mapping from component count k → reconstructed face vectors.
    indices : list of int
        Indices of faces to display.
    """
    ks = sorted(reconstructed_dict.keys())
    n_faces = len(indices)
    n_cols  = len(ks) + 1       # +1 for the original
    fig, axes = plt.subplots(n_faces, n_cols, figsize=(2.2 * n_cols, 2.5 * n_faces))
    if n_faces == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        # Original
        axes[row, 0].imshow(X_original[idx].reshape(shape), cmap=CMAP)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=FONT_SIZE_TICK)
        axes[row, 0].axis("off")

        # Reconstructions
        for col, k in enumerate(ks, start=1):
            axes[row, col].imshow(reconstructed_dict[k][idx].reshape(shape), cmap=CMAP)
            if row == 0:
                axes[row, col].set_title(f"k = {k}", fontsize=FONT_SIZE_TICK)
            axes[row, col].axis("off")

    fig.suptitle("Face Reconstruction with Increasing Components",
                 fontsize=FONT_SIZE_TITLE + 1, y=1.02)
    fig.tight_layout()
    _save(fig, "07_reconstruction_grid.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Reconstruction error vs k
# ═══════════════════════════════════════════════════════════════════════════

def plot_reconstruction_error(ks, mses):
    """
    Plot average reconstruction MSE as a function of the number of
    principal components used.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, mses, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Reconstruction Error vs. Number of Components")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "08_reconstruction_error.png")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Accuracy vs k
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_k(ks, accuracies, best_k=None, best_acc=None):
    """
    Plot classification accuracy as a function of the number of
    principal components used.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, accuracies, "s-", color="#4C72B0", linewidth=2, markersize=6)
    if best_k is not None and best_acc is not None:
        ax.axvline(best_k, color="#DD8452", linestyle="--", linewidth=1.2, label=f"Best k = {best_k}")
        ax.axhline(best_acc, color="#DD8452", linestyle=":", linewidth=1.0)
        ax.scatter([best_k], [best_acc], color="#DD8452", s=100, zorder=5,
                   label=f"Best accuracy = {best_acc:.2%}")
        ax.legend(fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Recognition Accuracy vs. Number of Components (1-NN)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "09_accuracy_vs_k.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Correctly classified examples
# ═══════════════════════════════════════════════════════════════════════════

def plot_correctly_classified(X_test, y_test, y_pred, n=10, shape=IMAGE_SHAPE):
    """
    Grid of correctly classified test faces with true labels.
    """
    correct = np.where(y_test == y_pred)[0]
    n = min(n, len(correct))
    if n == 0:
        print("  ⚠ No correctly classified samples to display.")
        return
    indices = correct[:n]

    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.4 * n_rows))
    fig.suptitle("Correctly Classified Faces", fontsize=FONT_SIZE_TITLE + 1, y=1.02)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(X_test[indices[i]].reshape(shape), cmap=CMAP)
            ax.set_title(f"True: {y_test[indices[i]]}", fontsize=FONT_SIZE_TICK, color="green")
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "10_correctly_classified.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. Misclassified examples
# ═══════════════════════════════════════════════════════════════════════════

def plot_misclassified(X_test, y_test, y_pred, n=10, shape=IMAGE_SHAPE):
    """
    Grid of misclassified test faces with true and predicted labels.
    """
    wrong = np.where(y_test != y_pred)[0]
    n = min(n, len(wrong))
    if n == 0:
        print("  ⚠ No misclassified samples to display.")
        return
    indices = wrong[:n]

    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows))
    fig.suptitle("Misclassified Faces", fontsize=FONT_SIZE_TITLE + 1, y=1.02)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(X_test[indices[i]].reshape(shape), cmap=CMAP)
            ax.set_title(f"True: {y_test[indices[i]]}  Pred: {y_pred[indices[i]]}",
                         fontsize=FONT_SIZE_TICK - 1, color="red")
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "11_misclassified.png")
