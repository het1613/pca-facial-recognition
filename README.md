# Face Recognition with PCA (Eigenfaces)

A modular Python project demonstrating Principal Component Analysis (PCA) applied to face recognition (commonly known as the Eigenfaces method). 

---

## Overview

PCA finds an orthonormal basis of eigenfaces that captures the directions of maximum variance in a set of training face images. Each face can then be represented as a compact linear combination of these eigenfaces, enabling efficient dimensionality reduction, reconstruction, and classification.

---

## Dataset

**Olivetti Faces** (AT&T / Cambridge)  
Loaded via `sklearn.datasets.fetch_olivetti_faces`.

| Property         | Value            |
|------------------|------------------|
| Subjects         | 40               |
| Images / subject | 10               |
| Total images     | 400              |
| Image size       | 64 × 64 pixels   |
| Pixel range      | [0, 1] (float)   |

The dataset is automatically downloaded and cached by scikit-learn on first run.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

All figures are saved to `output/figures/`.  
A summary of classification results is printed to the console.