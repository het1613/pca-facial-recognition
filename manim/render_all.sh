#!/usr/bin/env bash
# render_all.sh — Render every Manim scene at 1080p quality.
#
# Scenes are listed in logical presentation order:
#   1. Data representation      scene1_face_to_vector.py
#   2. Preprocessing             scene_mean_centering.py
#   3. PCA intuition (2-D)       scene_pca_intuition.py
#   4. LA — covariance & eigen   scene_covariance_eigen.py  (incl. Lagrangian derivation)
#   5. Eigenfaces basis          scene_eigenfaces.py
#   6. Variance explained        scene_variance_explained.py
#   7. Reconstruction            scene_reconstruction.py
#   8. Recognition (1-NN)        scene_recognition.py
#
# Usage:
#   cd manim/
#   bash render_all.sh
#
# Videos are saved under  manim/media/videos/*/1080p60/

set -euo pipefail

QUALITY="-qh"   # 1080p @ 60 fps  (use -ql for fast preview)

SCENES=(
    "scene1_face_to_vector.py    FaceToVectorScene"
    "scene_mean_centering.py     MeanCenteringScene"
    "scene_pca_intuition.py      PCAIntuition2DScene"
    "scene_covariance_eigen.py   CovarianceEigenScene"
    "scene_eigenfaces.py         EigenfacesScene"
    "scene_variance_explained.py VarianceExplainedScene"
    "scene_reconstruction.py     ReconstructionScene"
    "scene_recognition.py        RecognitionScene"
)

echo "═══════════════════════════════════════════════════════"
echo "  Rendering all Manim scenes ($QUALITY)"
echo "═══════════════════════════════════════════════════════"

for entry in "${SCENES[@]}"; do
    file=$(echo "$entry"  | awk '{print $1}')
    scene=$(echo "$entry" | awk '{print $2}')
    echo ""
    echo "▸ Rendering $scene  ($file)"
    manim render $QUALITY "$file" "$scene"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All scenes rendered ✓"
echo "═══════════════════════════════════════════════════════"
