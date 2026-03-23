"""
Scene 8 - CovarianceEigenScene

Explains the mathematical connection between the covariance matrix, its
eigenvectors (eigenfaces), and eigenvalues (explained variance).

NEW: adds the Lagrangian optimisation derivation to show *why* the
eigenvalue equation Cv = λv arises - not just that it does.

Flow:
  Screen 1  ->  Centred data matrix X̃  +  covariance  C = (1/N) X̃ᵀ X̃
  Screen 2  ->  Optimisation problem  ->  Lagrangian  ->  Cv = λv
  Screen 3  ->  Eigendecomposition interpretation  +  SVD connection
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

        # ==============================================================
        # SCREEN 1 - Centred data matrix + covariance matrix
        # ==============================================================

        step1 = styled_text("1. Centred data matrix", font_size=24, color=BLUE)
        step1.to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.8)
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
        X_mat.next_to(step1, DOWN, buff=0.25)

        X_dim = MathTex(r"\in \mathbb{R}^{N \times d}", font_size=26, color=DIM)
        X_dim.next_to(X_mat, RIGHT, buff=0.3)

        self.play(Write(X_mat), FadeIn(X_dim), run_time=1.2)
        self.wait(0.8)

        # -- Step 2: Covariance matrix ---------------------------------
        step2 = styled_text("2. Covariance matrix", font_size=24, color=BLUE)
        step2.next_to(X_mat, DOWN, buff=0.4).to_edge(LEFT, buff=0.8)
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
        self.wait(1.0)

        # Clear Screen 1
        self.play(
            FadeOut(step1), FadeOut(X_mat), FadeOut(X_dim),
            FadeOut(step2), FadeOut(cov_eq), FadeOut(cov_dim),
        )
        self.wait(0.2)

        # ==============================================================
        # SCREEN 2 - The Optimisation Problem -> Lagrangian -> Cv = λv
        # ==============================================================

        opt_header = styled_text(
            "3. The Optimisation Problem", font_size=24, color=BLUE,
        )
        opt_header.to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(opt_header))

        # Goal statement
        goal_text = styled_text(
            "Find the direction  v  that maximises variance of projected data:",
            font_size=22, color=TEXT_C,
        )
        goal_text.next_to(opt_header, DOWN, buff=0.25).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(goal_text))
        self.wait(0.3)

        # Optimisation problem
        opt_eq = MathTex(
            r"\max_{\mathbf{v}}",
            r"\; \mathbf{v}^T C \mathbf{v}",
            r"\quad \text{s.t.} \quad",
            r"\mathbf{v}^T \mathbf{v} = 1",
            font_size=32, color=TEXT_C,
        )
        opt_eq.set_color_by_tex(r"\mathbf{v}^T C \mathbf{v}", YELLOW)
        opt_eq.set_color_by_tex(r"C", GREEN)
        opt_eq.set_color_by_tex(r"\mathbf{v}^T \mathbf{v} = 1", BLUE)
        opt_eq.next_to(goal_text, DOWN, buff=0.25)
        self.play(Write(opt_eq), run_time=1.2)
        self.wait(0.8)

        # Lagrangian
        lag_label = styled_text("Lagrangian:", font_size=22, color=DIM)
        lag_label.next_to(opt_eq, DOWN, buff=0.3).to_edge(LEFT, buff=1.2)

        lag_eq = MathTex(
            r"\mathcal{L}(\mathbf{v}, \lambda)",
            r"=",
            r"\mathbf{v}^T C \mathbf{v}",
            r"-",
            r"\lambda \,(\mathbf{v}^T \mathbf{v} - 1)",
            font_size=28, color=TEXT_C,
        )
        lag_eq.set_color_by_tex(r"\mathbf{v}^T C \mathbf{v}", YELLOW)
        lag_eq.set_color_by_tex(r"\lambda", PEACH)
        lag_eq.next_to(lag_label, RIGHT, buff=0.4)

        self.play(FadeIn(lag_label), Write(lag_eq), run_time=1.1)
        self.wait(0.6)

        # Gradient condition
        grad_label = styled_text(
            "Set  ∇L = 0 :", font_size=22, color=DIM,
        )
        grad_label.next_to(lag_eq, DOWN, buff=0.28).to_edge(LEFT, buff=1.2)

        grad_eq = MathTex(
            r"\nabla_{\mathbf{v}} \mathcal{L}",
            r"=",
            r"2C\mathbf{v} - 2\lambda\mathbf{v}",
            r"=",
            r"\mathbf{0}",
            font_size=28, color=TEXT_C,
        )
        grad_eq.set_color_by_tex(r"2C\mathbf{v} - 2\lambda\mathbf{v}", TEXT_C)
        grad_eq.next_to(grad_label, RIGHT, buff=0.4)

        self.play(FadeIn(grad_label), Write(grad_eq), run_time=1.0)
        self.wait(0.5)

        # Key result - Cv = λv
        result_eq = MathTex(
            r"\Longrightarrow",
            r"\quad C\mathbf{v}",
            r"=",
            r"\lambda\mathbf{v}",
            font_size=36, color=PEACH,
        )
        result_eq.set_color_by_tex(r"C\mathbf{v}", GREEN)
        result_eq.set_color_by_tex(r"\lambda\mathbf{v}", YELLOW)
        result_eq.next_to(grad_eq, DOWN, buff=0.3).shift(RIGHT * 0.3)

        result_box = SurroundingRectangle(result_eq, color=PEACH, buff=0.2, corner_radius=0.1)

        self.play(Write(result_eq), run_time=0.9)
        self.play(Create(result_box), run_time=0.5)
        self.wait(1.5)

        # Clear Screen 2
        self.play(
            FadeOut(opt_header), FadeOut(goal_text), FadeOut(opt_eq),
            FadeOut(lag_label), FadeOut(lag_eq),
            FadeOut(grad_label), FadeOut(grad_eq),
            FadeOut(result_eq), FadeOut(result_box),
        )
        self.wait(0.2)

        # ==============================================================
        # SCREEN 3 - Eigendecomposition interpretation + SVD connection
        # ==============================================================

        step_eigen = styled_text("4. Eigendecomposition", font_size=24, color=BLUE)
        step_eigen.to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(step_eigen))

        eigen_eq = MathTex(
            r"C", r"\mathbf{v}_i", r"=", r"\lambda_i", r"\mathbf{v}_i",
            font_size=42, color=TEXT_C,
        )
        eigen_eq.set_color_by_tex("C", GREEN)
        eigen_eq.set_color_by_tex(r"\mathbf{v}_i", BLUE)
        eigen_eq.set_color_by_tex(r"\lambda_i", YELLOW)
        eigen_eq.move_to(UP * 1.2)

        self.play(Write(eigen_eq), run_time=1.0)
        self.wait(0.5)

        # Interpretation bullets
        interp1 = VGroup(
            MathTex(r"\mathbf{v}_i", font_size=30, color=BLUE),
            styled_text(
                " -> eigenface (eigenvector, reshaped to 64 × 64)",
                font_size=22, color=TEXT_C,
            ),
        ).arrange(RIGHT, buff=0.15)

        interp2 = VGroup(
            MathTex(r"\lambda_i", font_size=30, color=YELLOW),
            styled_text(
                " -> variance captured along direction  v_i",
                font_size=22, color=TEXT_C,
            ),
        ).arrange(RIGHT, buff=0.15)

        interp3 = VGroup(
            MathTex(r"\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}", font_size=28, color=LAVEN),
            styled_text(
                "  - eigenvectors are orthonormal  ->  clean projection & reconstruction",
                font_size=22, color=TEXT_C,
            ),
        ).arrange(RIGHT, buff=0.15)

        interp = VGroup(interp1, interp2, interp3).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        interp.move_to(DOWN * 0.3)

        self.play(FadeIn(interp, shift=UP * 0.2), run_time=1.0)
        self.wait(1.0)

        # -- SVD connection (numerics) ----------------------------------
        svd_note = styled_text(
            "In practice - use SVD of  X̃  (avoids forming 4096 × 4096 covariance matrix):",
            font_size=19, color=DIM,
        )
        svd_note.to_edge(DOWN, buff=0.5)

        svd_eq = MathTex(
            r"\tilde{X} = U \Sigma V^T",
            r"\quad \Rightarrow \quad",
            r"\text{eigenfaces} = \text{columns of } V,",
            r"\quad \lambda_i = \tfrac{\sigma_i^2}{N-1}",
            font_size=24, color=LAVEN,
        )
        svd_eq.set_color_by_tex(r"\lambda_i", YELLOW)
        svd_eq.next_to(svd_note, UP, buff=0.2)

        self.play(FadeIn(svd_note), Write(svd_eq), run_time=1.2)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
