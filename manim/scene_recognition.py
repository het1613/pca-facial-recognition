"""
Scene 6 - RecognitionScene

Demonstrates nearest-neighbour face recognition in PCA space:
  1. Show a 2-D PCA scatter plot of training faces, colour-coded by identity.
  2. Animate a test face being projected into this space.
  3. Draw a line to its nearest neighbour -> classification.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM, LAVEN,
    CLASS_COLORS, section_title, styled_text, math,
)


class RecognitionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # -- Prepare data ----------------------------------------------
        data = fetch_olivetti_faces(shuffle=False)
        X, y = data.data, data.target
        # Use a subset of subjects for visual clarity
        subjects = list(range(8))
        mask = np.isin(y, subjects)
        X_sub, y_sub = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.25, random_state=42, stratify=y_sub,
        )
        mean_face = X_train.mean(axis=0)
        X_train_c = X_train - mean_face
        X_test_c  = X_test  - mean_face

        pca = PCA(n_components=50)
        pca.fit(X_train_c)
        X_train_2d = X_train_c @ pca.components_[:2].T
        X_test_2d  = X_test_c  @ pca.components_[:2].T

        # -- Build axes -------------------------------------------------
        all_pts = np.vstack([X_train_2d, X_test_2d])
        pad = 1.5
        x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
        y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

        axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=8, y_length=5,
            tips=False,
            axis_config={"color": DIM, "stroke_width": 1.2},
        ).shift(UP * 0.1)
        ax_labels = axes.get_axis_labels(
            MathTex("w_1", font_size=24, color=DIM),
            MathTex("w_2", font_size=24, color=DIM),
        )
        self.play(Create(axes), FadeIn(ax_labels), run_time=0.7)

        # -- Plot training points (colour-coded) -----------------------
        legend_items = VGroup()
        train_dots = VGroup()
        for subj in subjects:
            mask_s = y_train == subj
            color = CLASS_COLORS[subj % len(CLASS_COLORS)]
            for pt in X_train_2d[mask_s]:
                dot = Dot(axes.c2p(pt[0], pt[1]), radius=0.055,
                          color=color, fill_opacity=0.8)
                train_dots.add(dot)
            # Legend entry
            leg_dot = Dot(radius=0.06, color=color)
            leg_text = styled_text(f"S{subj}", font_size=16, color=color)
            entry = VGroup(leg_dot, leg_text).arrange(RIGHT, buff=0.1)
            legend_items.add(entry)

        legend_items.arrange_in_grid(rows=2, buff=(0.3, 0.15))
        legend_items.to_corner(UR, buff=0.4)

        self.play(FadeIn(train_dots, lag_ratio=0.01), FadeIn(legend_items), run_time=1.0)
        self.wait(0.5)

        train_label = styled_text("Training faces (projected)", font_size=20, color=DIM)
        train_label.next_to(axes, DOWN, buff=0.1)
        self.play(FadeIn(train_label))
        self.wait(0.5)

        # -- Animate a test face entering the space --------------------
        self.play(FadeOut(train_label))

        test_idx = 0
        test_pt = X_test_2d[test_idx]
        test_label_true = y_test[test_idx]

        test_dot = Dot(axes.c2p(test_pt[0], test_pt[1]), radius=0.09,
                       color=YELLOW, fill_opacity=0)
        test_star = Star(n=5, outer_radius=0.15, inner_radius=0.07,
                         color=YELLOW, fill_opacity=1)
        test_star.move_to(axes.c2p(test_pt[0], test_pt[1]))

        test_text = styled_text("Test face (unknown)", font_size=20, color=YELLOW)
        test_text.next_to(test_star, UP, buff=0.2)

        # Fly in from the side
        test_star_start = test_star.copy().move_to(axes.c2p(x_max + 2, test_pt[1]))
        self.play(
            test_star.animate.move_to(axes.c2p(test_pt[0], test_pt[1])),
            FadeIn(test_text),
            run_time=1.2,
        )
        self.wait(0.5)

        # -- Find nearest neighbour ------------------------------------
        dists = np.linalg.norm(X_train_2d - test_pt, axis=1)
        nn_idx = np.argmin(dists)
        nn_pt = X_train_2d[nn_idx]
        nn_label = y_train[nn_idx]
        nn_color = CLASS_COLORS[nn_label % len(CLASS_COLORS)]

        nn_line = DashedLine(
            axes.c2p(test_pt[0], test_pt[1]),
            axes.c2p(nn_pt[0], nn_pt[1]),
            color=PEACH, stroke_width=2.5,
        )
        nn_text = styled_text("Nearest neighbour", font_size=18, color=PEACH)
        nn_text.next_to(nn_line.get_center(), RIGHT, buff=0.15)

        self.play(Create(nn_line), FadeIn(nn_text), run_time=1.0)
        self.wait(0.5)

        # -- Classification result -------------------------------------
        correct = test_label_true == nn_label
        result_color = GREEN if correct else PEACH
        result_symbol = "" if correct else "✗"
        result_text = styled_text(
            f"Predicted: Subject {nn_label}  {result_symbol}",
            font_size=24, color=result_color,
        )
        result_text.to_edge(DOWN, buff=0.35)

        self.play(FadeOut(test_text), FadeIn(result_text))
        self.wait(0.6)

        # -- Formula ---------------------------------------------------
        formula = MathTex(
            r"\hat{y} = y_{\arg\min_i \| w_{\text{test}} - w_i \|_2}",
            font_size=30, color=LAVEN,
        )
        formula.next_to(result_text, UP, buff=0.2)
        self.play(Write(formula))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
