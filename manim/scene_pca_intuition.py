"""
Scene 3 - PCAIntuition2DScene

Uses a toy 2-D correlated point cloud to build geometric intuition for PCA:
  1. Show scattered data points.
  2. Animate PC1 - the direction of maximum variance.
  3. Project points onto PC1.
  4. Show PC2 - orthogonal to PC1.
  5. Explain "maximum variance" intuition.
"""

from manim import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM, LAVEN,
    section_title, styled_text, math,
)


class PCAIntuition2DScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # -- Generate correlated 2-D data ------------------------------
        np.random.seed(42)
        n = 60
        # Correlated Gaussian
        angle = np.pi / 6          # ~30° tilt
        cov = np.array([[2.5, 1.5],
                        [1.5, 1.0]])
        mean = np.array([0, 0])
        pts = np.random.multivariate_normal(mean, cov, n)

        # PCA by hand
        pts_centered = pts - pts.mean(axis=0)
        C = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        pc1 = eigvecs[:, 0]
        pc2 = eigvecs[:, 1]

        # Axes
        axes = Axes(
            x_range=[-5, 5, 1], y_range=[-4, 4, 1],
            x_length=8, y_length=5.5,
            tips=False,
            axis_config={"color": DIM, "stroke_width": 1.5},
        ).shift(UP * 0.2)

        ax_labels = axes.get_axis_labels(
            MathTex("x_1", font_size=26, color=DIM),
            MathTex("x_2", font_size=26, color=DIM),
        )

        self.play(Create(axes), FadeIn(ax_labels), run_time=0.8)

        # -- Plot data points ------------------------------------------
        dots = VGroup()
        for p in pts_centered:
            dot = Dot(axes.c2p(p[0], p[1]), radius=0.045, color=BLUE, fill_opacity=0.75)
            dots.add(dot)

        self.play(FadeIn(dots, lag_ratio=0.02), run_time=1.0)
        self.wait(0.5)

        # -- Data cloud label ------------------------------------------
        cloud_label = styled_text("Centered data cloud", font_size=22, color=DIM)
        cloud_label.next_to(axes, DOWN, buff=0.1)
        self.play(FadeIn(cloud_label))
        self.wait(0.5)

        # -- PC1: direction of maximum variance ------------------------
        pc1_label_text = styled_text("PC 1 - max variance direction", font_size=22, color=PEACH)
        pc1_label_text.to_edge(DOWN, buff=0.4)

        scale = 4.0
        pc1_line = Line(
            axes.c2p(*(- scale * pc1)),
            axes.c2p(*(  scale * pc1)),
            color=PEACH, stroke_width=3,
        )
        pc1_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(*(2.5 * pc1)),
            color=PEACH, stroke_width=4, buff=0,
            max_tip_length_to_length_ratio=0.12,
        )
        pc1_tex = MathTex(r"\mathbf{v}_1", font_size=28, color=PEACH)
        pc1_tex.next_to(pc1_arrow.get_end(), UR, buff=0.1)

        self.play(
            FadeOut(cloud_label),
            Create(pc1_line, run_time=1.0),
            GrowArrow(pc1_arrow),
            FadeIn(pc1_tex),
            FadeIn(pc1_label_text),
        )
        self.wait(0.8)

        # -- Project points onto PC1 -----------------------------------
        proj_dots = VGroup()
        proj_lines = VGroup()
        for p in pts_centered:
            proj_scalar = np.dot(p, pc1)
            proj_pt = proj_scalar * pc1
            proj_dot = Dot(axes.c2p(*proj_pt), radius=0.04, color=YELLOW, fill_opacity=0.8)
            proj_line = DashedLine(
                axes.c2p(*p), axes.c2p(*proj_pt),
                color=YELLOW, stroke_width=0.8, stroke_opacity=0.4,
            )
            proj_dots.add(proj_dot)
            proj_lines.add(proj_line)

        self.play(
            FadeIn(proj_lines, lag_ratio=0.01),
            FadeIn(proj_dots, lag_ratio=0.01),
            run_time=1.5,
        )
        self.wait(1.0)

        # -- Remove projections, show PC2 ------------------------------
        self.play(FadeOut(proj_lines), FadeOut(proj_dots), FadeOut(pc1_label_text), run_time=0.6)

        pc2_line = Line(
            axes.c2p(*(- scale * pc2)),
            axes.c2p(*(  scale * pc2)),
            color=GREEN, stroke_width=2.5,
        )
        pc2_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(*(2.0 * pc2)),
            color=GREEN, stroke_width=4, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        pc2_tex = MathTex(r"\mathbf{v}_2", font_size=28, color=GREEN)
        pc2_tex.next_to(pc2_arrow.get_end(), UL, buff=0.1)

        perp_label = styled_text("PC 2 - orthogonal, next most variance", font_size=22, color=GREEN)
        perp_label.to_edge(DOWN, buff=0.4)

        self.play(
            Create(pc2_line), GrowArrow(pc2_arrow),
            FadeIn(pc2_tex), FadeIn(perp_label),
            run_time=1.0,
        )
        self.wait(0.8)

        # -- Orthogonality callout --------------------------------------
        ortho = MathTex(r"\mathbf{v}_1 \perp \mathbf{v}_2", font_size=30, color=LAVEN)
        ortho.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)
        angle_arc = Angle(pc1_line, pc2_line, radius=0.6, color=LAVEN)

        self.play(Write(ortho), Create(angle_arc), run_time=0.8)
        self.wait(1.5)

        # -- Variance explanation --------------------------------------
        self.play(FadeOut(perp_label))
        var_text = MathTex(
            r"\text{Var along } \mathbf{v}_1 > \text{Var along } \mathbf{v}_2",
            font_size=28, color=YELLOW,
        )
        var_text.to_edge(DOWN, buff=0.4)
        self.play(Write(var_text))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
