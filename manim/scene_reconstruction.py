"""
Scene 5 — ReconstructionScene

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


class ReconstructionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Compute PCA ────────────────────────────────────────────────
        data = fetch_olivetti_faces(shuffle=False)
        X = data.data
        mean_face = X.mean(axis=0)
        X_centered = X - mean_face
        n_max = 300
        pca = PCA(n_components=n_max)
        pca.fit(X_centered)

        face_idx = 5                          # pick a face to reconstruct
        original = X[face_idx]
        x_centered = X_centered[face_idx]

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("Face Reconstruction with Eigenfaces")
        self.play(Write(title))
        self.wait(0.4)

        # ── Reconstruction formula ─────────────────────────────────────
        formula = MathTex(
            r"\hat{x}", r"\approx", r"\mu", r"+",
            r"\sum_{i=1}^{k}", r"a_i", r"\mathbf{u}_i",
            font_size=34, color=TEXT_C,
        )
        formula.set_color_by_tex(r"\mu", PEACH)
        formula.set_color_by_tex(r"a_i", YELLOW)
        formula.set_color_by_tex(r"\mathbf{u}_i", BLUE)
        formula.next_to(title, DOWN, buff=0.35)
        self.play(Write(formula))
        self.wait(0.6)

        # ── Original face (left) ──────────────────────────────────────
        orig_thumb = create_face_thumbnail(original.reshape(64, 64), height=2.2)
        orig_thumb.move_to(LEFT * 4.5 + DOWN * 1.0)
        orig_label = styled_text("Original", font_size=20, color=GREEN)
        orig_label.next_to(orig_thumb, DOWN, buff=0.2)
        self.play(FadeIn(orig_thumb), FadeIn(orig_label))

        # ── Arrow ─────────────────────────────────────────────────────
        arrow = Arrow(LEFT * 2.8 + DOWN * 1.0, LEFT * 1.2 + DOWN * 1.0,
                      color=PEACH, stroke_width=3)
        self.play(GrowArrow(arrow))

        # ── Progressive reconstruction (right, replacing in place) ────
        ks = [1, 5, 10, 20, 50, 100, 200, 300]
        recon_pos = RIGHT * 1.5 + DOWN * 1.0

        prev_thumb = None
        prev_label_mob = None

        for k in ks:
            V_k = pca.components_[:k]
            weights = x_centered @ V_k.T
            recon = mean_face + weights @ V_k
            recon_clipped = np.clip(recon, 0, 1)

            thumb = create_face_thumbnail(recon_clipped.reshape(64, 64), height=2.2)
            thumb.move_to(recon_pos)
            k_label = styled_text(f"k = {k}", font_size=22, color=YELLOW)
            k_label.next_to(thumb, DOWN, buff=0.2)

            # Compute MSE
            mse = np.mean((original - recon_clipped) ** 2)
            mse_label = styled_text(f"MSE = {mse:.4f}", font_size=18, color=DIM)
            mse_label.next_to(k_label, DOWN, buff=0.1)

            if prev_thumb is None:
                self.play(FadeIn(thumb), FadeIn(k_label), FadeIn(mse_label), run_time=0.8)
            else:
                self.play(
                    FadeOut(prev_thumb), FadeOut(prev_label_mob), FadeOut(prev_mse),
                    FadeIn(thumb), FadeIn(k_label), FadeIn(mse_label),
                    run_time=0.7,
                )
            self.wait(0.5)

            prev_thumb = thumb
            prev_label_mob = k_label
            prev_mse = mse_label

        # ── Final comparison ──────────────────────────────────────────
        compare = styled_text(
            "More components → better reconstruction (lower MSE)",
            font_size=22, color=GREEN,
        )
        compare.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(compare))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
