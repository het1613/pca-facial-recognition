"""
Scene 4 — EigenfacesScene

Displays the mean face followed by the top eigenfaces appearing one-by-one, with labels and explanatory text.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM, LAVEN,
    section_title, styled_text, math, create_face_thumbnail,
)


class EigenfacesScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Compute eigenfaces from Olivetti data ─────────────────────
        data = fetch_olivetti_faces(shuffle=False)
        X = data.data                              # (400, 4096)
        mean_face = X.mean(axis=0)
        X_centered = X - mean_face
        pca = PCA(n_components=100)
        pca.fit(X_centered)
        eigenfaces = pca.components_.reshape(-1, 64, 64)   # (100, 64, 64)

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("Eigenfaces — Basis of Facial Variation")
        self.play(Write(title))
        self.wait(0.4)

        # ── Mean face ─────────────────────────────────────────────────
        mean_img = mean_face.reshape(64, 64)
        mean_thumb = create_face_thumbnail(mean_img, height=2.0)
        mean_thumb.move_to(LEFT * 4.5 + DOWN * 0.2)
        mean_label = MathTex(r"\mu", font_size=34, color=PEACH)
        mean_label.next_to(mean_thumb, DOWN, buff=0.2)
        mean_caption = styled_text("Mean Face", font_size=20, color=DIM)
        mean_caption.next_to(mean_label, DOWN, buff=0.1)

        self.play(FadeIn(mean_thumb, shift=UP * 0.3), FadeIn(mean_label), FadeIn(mean_caption))
        self.wait(0.6)

        # ── Plus sign ─────────────────────────────────────────────────
        plus = MathTex(r"+", font_size=40, color=TEXT_C).next_to(mean_thumb, RIGHT, buff=0.4)
        self.play(FadeIn(plus))

        # ── Eigenfaces appearing one by one (show first 8) ────────────
        n_show = 8
        n_cols = 4
        n_rows = 2

        ef_group = Group()
        ef_labels = VGroup()

        for i in range(n_show):
            ef = eigenfaces[i]
            # Normalise for display
            emin, emax = ef.min(), ef.max()
            if emax - emin > 0:
                ef_disp = (ef - emin) / (emax - emin)
            else:
                ef_disp = ef * 0 + 0.5
            thumb = create_face_thumbnail(ef_disp, height=1.3)
            label = MathTex(f"u_{{{i+1}}}", font_size=22, color=BLUE)
            ef_group.add(thumb)
            ef_labels.add(label)

        # Arrange in grid
        rows = Group()
        for r in range(n_rows):
            row = Group(*ef_group[r * n_cols : (r + 1) * n_cols])
            row.arrange(RIGHT, buff=0.4)
            rows.add(row)
        rows.arrange(DOWN, buff=0.5)
        rows.next_to(plus, RIGHT, buff=0.5)

        # Position labels below each thumbnail
        for lbl, thumb in zip(ef_labels, ef_group):
            lbl.next_to(thumb, DOWN, buff=0.1)

        # Animate one by one
        for i in range(n_show):
            self.play(
                FadeIn(ef_group[i], shift=DOWN * 0.15),
                FadeIn(ef_labels[i]),
                run_time=0.5,
            )
        self.wait(0.5)

        # ── Explanatory text ──────────────────────────────────────────
        explain1 = styled_text(
            "Eigenfaces = eigenvectors of the covariance matrix",
            font_size=22, color=GREEN,
        )
        explain2 = styled_text(
            "They form an orthonormal basis for facial variation",
            font_size=22, color=GREEN,
        )
        explain_group = VGroup(explain1, explain2).arrange(DOWN, buff=0.15)
        explain_group.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(explain_group, shift=UP * 0.2))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
