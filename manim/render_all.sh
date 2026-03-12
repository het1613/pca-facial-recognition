#!/usr/bin/env bash
# render_all.sh — Render every Manim scene at 1080p quality.
#
# Usage:
#   cd manim/
#   bash render_all.sh
#
# Videos are saved under  manim/media/videos/*/1080p60/

set -euo pipefail

QUALITY="-qh"   # 1080p @ 60 fps  (use -ql for fast preview)

SCENES=(
    "scene_face_to_vector.py   FaceToVectorScene"
    "scene_mean_centering.py   MeanCenteringScene"
    "scene_pca_intuition.py    PCAIntuition2DScene"
    "scene_eigenfaces.py       EigenfacesScene"
    "scene_reconstruction.py   ReconstructionScene"
    "scene_recognition.py      RecognitionScene"
    "scene_variance_explained.py VarianceExplainedScene"
    "scene_covariance_eigen.py CovarianceEigenScene"
)

echo "═══════════════════════════════════════════════════════"
echo "  Rendering all Manim scenes ($QUALITY)"
echo "═══════════════════════════════════════════════════════"

for entry in "${SCENES[@]}"; do
    file=$(echo "$entry" | awk '{print $1}')
    scene=$(echo "$entry" | awk '{print $2}')
    echo ""
    echo "▸ Rendering $scene  ($file)"
    manim render $QUALITY "$file" "$scene"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All scenes rendered ✓"
echo "═══════════════════════════════════════════════════════"
