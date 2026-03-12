"""
Scene 7 — VarianceExplainedScene

Animates the cumulative explained variance bar/line chart as more
principal components are added, marking the 90 % and 95 % thresholds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM, LAVEN,
    section_title, styled_text, math,
)


class VarianceExplainedScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Compute PCA ────────────────────────────────────────────────
        data = fetch_olivetti_faces(shuffle=False)
        X = data.data
        X_centered = X - X.mean(axis=0)
        pca = PCA(n_components=100)
        pca.fit(X_centered)
        cum_var = np.cumsum(pca.explained_variance_ratio_)  # length 100

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("Cumulative Explained Variance")
        self.play(Write(title))
        self.wait(0.4)

        # ── Build axes ─────────────────────────────────────────────────
        n_bars = 40           # show first 40 components for clarity
        axes = Axes(
            x_range=[0, n_bars + 1, 5],
            y_range=[0, 1.05, 0.2],
            x_length=9,
            y_length=4.5,
            tips=False,
            axis_config={"color": DIM, "stroke_width": 1.2,
                         "include_numbers": True, "font_size": 20},
        ).shift(DOWN * 0.4)

        x_label = MathTex("k", font_size=24, color=DIM).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = styled_text("Cumulative variance", font_size=18, color=DIM)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3).shift(UP * 0.5)
        y_label.rotate(PI / 2)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.8)

        # ── Animate bars growing one by one ───────────────────────────
        bars = VGroup()
        bar_width = 0.18

        # Build all bars first (invisible)
        for i in range(n_bars):
            x_pos = axes.c2p(i + 1, 0)[0]
            y_top = axes.c2p(0, cum_var[i])[1]
            y_bot = axes.c2p(0, 0)[1]
            height = y_top - y_bot
            bar = Rectangle(
                width=bar_width, height=height,
                fill_color=BLUE, fill_opacity=0.75,
                stroke_color=BLUE, stroke_width=0.5,
            )
            bar.move_to(np.array([x_pos, y_bot + height / 2, 0]))
            bars.add(bar)

        # Animate in three batches
        batch_sizes = [10, 15, 15]
        idx = 0
        for batch in batch_sizes:
            end = min(idx + batch, n_bars)
            self.play(
                *[GrowFromEdge(bars[i], DOWN) for i in range(idx, end)],
                run_time=1.0,
            )
            idx = end
        self.wait(0.4)

        # ── Threshold lines ───────────────────────────────────────────
        for thresh, color, ls in [(0.90, PEACH, "--"), (0.95, GREEN, ":")]:
            y_pos = axes.c2p(0, thresh)[1]
            line = DashedLine(
                np.array([axes.c2p(0, 0)[0], y_pos, 0]),
                np.array([axes.c2p(n_bars + 1, 0)[0], y_pos, 0]),
                color=color, stroke_width=1.5, dash_length=0.12,
            )
            # Find k where cumulative variance exceeds threshold
            k_thresh = int(np.searchsorted(cum_var, thresh)) + 1
            label = styled_text(
                f"{int(thresh*100)}% at k={k_thresh}",
                font_size=18, color=color,
            )
            label.next_to(line, RIGHT, buff=0.15).shift(UP * 0.15)
            self.play(Create(line), FadeIn(label), run_time=0.6)

        self.wait(0.5)

        # ── Explanatory text ──────────────────────────────────────────
        explain = styled_text(
            "A small number of components captures most of the variance",
            font_size=22, color=YELLOW,
        )
        explain.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(explain))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
