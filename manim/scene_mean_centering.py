"""
Scene 2 — MeanCenteringScene

Shows how the mean face is computed and subtracted from each training image, producing centered (zero-mean) face vectors.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM,
    section_title, styled_text, math, create_face_thumbnail,
)


class MeanCenteringScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Load faces ─────────────────────────────────────────────────
        data = fetch_olivetti_faces(shuffle=False)
        images = data.images                 # (400, 64, 64)
        n_show = 5
        face_indices = [0, 10, 20, 30, 40]  # one per subject
        face_imgs = [images[i] for i in face_indices]

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("Mean Face & Centering")
        self.play(Write(title))
        self.wait(0.5)

        # ── Step 1: show training faces ────────────────────────────────
        step1 = styled_text("Step 1: Training faces", font_size=24, color=BLUE)
        step1.next_to(title, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(step1))

        thumbs = Group()
        labels = VGroup()
        for i, img in enumerate(face_imgs):
            th = create_face_thumbnail(img, height=1.4)
            lbl = MathTex(f"x_{{{i+1}}}", font_size=26, color=TEXT_C)
            thumbs.add(th)
            labels.add(lbl)

        thumbs.arrange(RIGHT, buff=0.5)
        thumbs.next_to(step1, DOWN, buff=0.4)
        for lbl, th in zip(labels, thumbs):
            lbl.next_to(th, DOWN, buff=0.15)

        self.play(
            *[FadeIn(th, shift=UP * 0.2) for th in thumbs],
            *[FadeIn(lbl) for lbl in labels],
            run_time=1.2,
        )
        self.wait(0.8)

        # ── Step 2: compute mean ───────────────────────────────────────
        step2 = styled_text("Step 2: Compute mean face", font_size=24, color=BLUE)
        step2.next_to(thumbs, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)

        mean_formula = MathTex(
            r"\mu = \frac{1}{N} \sum_{i=1}^{N} x_i",
            font_size=32, color=YELLOW,
        )
        mean_formula.next_to(step2, RIGHT, buff=0.5)

        self.play(FadeIn(step2), Write(mean_formula), run_time=1.0)

        # Animate: all thumbs converge to center → mean face appears
        mean_img = np.mean(np.array(face_imgs), axis=0)
        mean_thumb = create_face_thumbnail(mean_img, height=1.6)
        mean_thumb.next_to(step2, DOWN, buff=0.4)

        mean_label = MathTex(r"\mu", font_size=32, color=PEACH)
        mean_label.next_to(mean_thumb, DOWN, buff=0.15)

        # Copies converge
        copies = Group(*[th.copy() for th in thumbs])
        self.play(
            *[c.animate.move_to(mean_thumb.get_center()).set_opacity(0.3) for c in copies],
            run_time=1.5,
        )
        self.play(FadeOut(copies), FadeIn(mean_thumb), FadeIn(mean_label), run_time=0.8)
        self.wait(0.6)

        # ── Step 3: subtract mean ──────────────────────────────────────
        # Fade out top items, shift everything up
        to_clear = Group(step1, thumbs, labels, step2, mean_formula)
        self.play(FadeOut(to_clear), run_time=0.6)

        # Move mean face to the left
        self.play(
            mean_thumb.animate.move_to(LEFT * 4.5 + UP * 0.5).scale(0.8),
            mean_label.animate.move_to(LEFT * 4.5 + DOWN * 0.6),
        )

        step3_title = styled_text("Step 3: Center the data", font_size=24, color=BLUE)
        step3_title.next_to(title, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(step3_title))

        formula = MathTex(
            r"\tilde{x}_i = x_i - \mu",
            font_size=36, color=YELLOW,
        )
        formula.move_to(UP * 0)
        self.play(Write(formula))
        self.wait(0.6)

        # Show centered faces (original - mean)
        centered_thumbs = Group()
        centered_labels = VGroup()
        for i, img in enumerate(face_imgs):
            centered = img - mean_img
            # Rescale for display
            cmin, cmax = centered.min(), centered.max()
            if cmax - cmin > 0:
                centered_disp = (centered - cmin) / (cmax - cmin)
            else:
                centered_disp = centered * 0 + 0.5
            ct = create_face_thumbnail(centered_disp, height=1.2)
            cl = MathTex(f"\\tilde{{x}}_{{{i+1}}}", font_size=24, color=TEXT_C)
            centered_thumbs.add(ct)
            centered_labels.add(cl)

        centered_thumbs.arrange(RIGHT, buff=0.4)
        centered_thumbs.move_to(DOWN * 1.8)
        for cl, ct in zip(centered_labels, centered_thumbs):
            cl.next_to(ct, DOWN, buff=0.12)

        self.play(
            *[FadeIn(ct, shift=DOWN * 0.2) for ct in centered_thumbs],
            *[FadeIn(cl) for cl in centered_labels],
            run_time=1.2,
        )

        explain = styled_text(
            "Centering removes shared structure → reveals individual variation",
            font_size=22, color=GREEN,
        )
        explain.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(explain))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
