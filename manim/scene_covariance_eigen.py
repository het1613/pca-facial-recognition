"""
Scene 8 — CovarianceEigenScene

Explains the mathematical connection between the covariance matrix, its eigenvectors (eigenfaces), and eigenvalues (explained variance).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from manim import *
import numpy as np

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM, LAVEN,
    section_title, styled_text, math,
)


class CovarianceEigenScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("From Covariance to Eigenfaces")
        self.play(Write(title))
        self.wait(0.5)

        # ── Step 1: Centered data matrix ──────────────────────────────
        step1 = styled_text("1. Centered data matrix", font_size=24, color=BLUE)
        step1.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(step1))

        X_mat = MathTex(
            r"\tilde{X}", r"=",
            r"\begin{bmatrix}"
            r"- \tilde{x}_1^T - \\"
            r"- \tilde{x}_2^T - \\"
            r"\vdots \\"
            r"- \tilde{x}_N^T -"
            r"\end{bmatrix}",
            font_size=30, color=TEXT_C,
        )
        X_mat.set_color_by_tex(r"\tilde{X}", PEACH)
        X_mat.next_to(step1, DOWN, buff=0.3)

        X_dim = MathTex(r"\in \mathbb{R}^{N \times d}", font_size=26, color=DIM)
        X_dim.next_to(X_mat, RIGHT, buff=0.3)

        self.play(Write(X_mat), FadeIn(X_dim), run_time=1.2)
        self.wait(0.8)

        # ── Step 2: Covariance matrix ─────────────────────────────────
        step2 = styled_text("2. Covariance matrix", font_size=24, color=BLUE)
        step2.next_to(X_mat, DOWN, buff=0.5).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(step2))

        cov_eq = MathTex(
            r"C", r"=", r"\frac{1}{N}", r"\tilde{X}^T", r"\tilde{X}",
            font_size=34, color=TEXT_C,
        )
        cov_eq.set_color_by_tex("C", GREEN)
        cov_eq.set_color_by_tex(r"\tilde{X}^T", PEACH)
        cov_eq.set_color_by_tex(r"\tilde{X}", PEACH)
        cov_eq.next_to(step2, RIGHT, buff=0.5)

        cov_dim = MathTex(r"\in \mathbb{R}^{d \times d}", font_size=26, color=DIM)
        cov_dim.next_to(cov_eq, RIGHT, buff=0.3)

        self.play(Write(cov_eq), FadeIn(cov_dim), run_time=1.0)
        self.wait(0.8)

        # ── Step 3: Eigendecomposition ────────────────────────────────
        # Clear top section
        self.play(
            FadeOut(step1), FadeOut(X_mat), FadeOut(X_dim),
            FadeOut(step2), FadeOut(cov_eq), FadeOut(cov_dim),
        )

        step3 = styled_text("3. Eigendecomposition", font_size=24, color=BLUE)
        step3.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(step3))

        eigen_eq = MathTex(
            r"C", r"\mathbf{v}_i", r"=", r"\lambda_i", r"\mathbf{v}_i",
            font_size=38, color=TEXT_C,
        )
        eigen_eq.set_color_by_tex("C", GREEN)
        eigen_eq.set_color_by_tex(r"\mathbf{v}_i", BLUE)
        eigen_eq.set_color_by_tex(r"\lambda_i", YELLOW)
        eigen_eq.move_to(UP * 0.5)

        self.play(Write(eigen_eq), run_time=1.2)
        self.wait(0.6)

        # ── Connect to eigenfaces ─────────────────────────────────────
        interp1 = VGroup(
            MathTex(r"\mathbf{v}_i", font_size=30, color=BLUE),
            styled_text(" → eigenface (reshaped to 64×64)", font_size=22, color=TEXT_C),
        ).arrange(RIGHT, buff=0.15)

        interp2 = VGroup(
            MathTex(r"\lambda_i", font_size=30, color=YELLOW),
            styled_text(" → variance captured along that direction", font_size=22, color=TEXT_C),
        ).arrange(RIGHT, buff=0.15)

        interp = VGroup(interp1, interp2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        interp.move_to(DOWN * 1.0)

        self.play(FadeIn(interp, shift=UP * 0.2), run_time=1.0)
        self.wait(1.0)

        # ── SVD connection ────────────────────────────────────────────
        svd_note = styled_text(
            "In practice: use SVD  X̃ = UΣVᵀ  (numerically stable)",
            font_size=20, color=DIM,
        )
        svd_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(svd_note))
        self.wait(0.6)

        svd_eq = MathTex(
            r"\tilde{X} = U \Sigma V^T",
            font_size=32, color=LAVEN,
        )
        svd_eq.next_to(svd_note, UP, buff=0.25)
        self.play(Write(svd_eq))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
