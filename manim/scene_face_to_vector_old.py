"""
Scene 1 — FaceToVectorScene

Demonstrates how a 2-D face image is flattened into a high-dimensional vector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np

# Add project root so we can load real faces
from sklearn.datasets import fetch_olivetti_faces

from helpers import (
    BG, TEXT_C, BLUE, PEACH, GREEN, YELLOW, DIM,
    section_title, styled_text, math, create_pixel_grid,
)


class FaceToVectorScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────
        title = section_title("Face Image → Vector Representation")
        self.play(Write(title))
        self.wait(0.5)

        # ── Load a real face and build an 8×8 pixel grid ──────────────
        faces = fetch_olivetti_faces(shuffle=False)
        face_img = faces.images[0]                       # (64, 64)

        grid = create_pixel_grid(face_img, grid_size=8, cell_size=0.4)
        grid.move_to(LEFT * 3.5)

        face_label = styled_text("Face image (8×8 shown)", font_size=22, color=DIM)
        face_label.next_to(grid, DOWN, buff=0.3)

        self.play(FadeIn(grid, shift=UP * 0.3), FadeIn(face_label), run_time=1.2)
        self.wait(0.8)

        # ── Arrow indicating transformation ───────────────────────────
        arrow = Arrow(LEFT * 1.2, RIGHT * 0.8, color=PEACH, stroke_width=4)
        arrow_label = styled_text("flatten", font_size=22, color=PEACH)
        arrow_label.next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(arrow_label))
        self.wait(0.3)

        # ── Build column vector representation ─────────────────────────
        # Show a stylised column vector with a few entries + dots
        from PIL import Image as PILImage
        pil = PILImage.fromarray((face_img * 255).astype(np.uint8), mode="L")
        pil = pil.resize((8, 8), PILImage.BILINEAR)
        small = np.array(pil).astype(float) / 255.0
        flat = small.flatten()

        entries = VGroup()
        n_show = 6                          # show first few + last few
        display_vals = list(flat[:n_show])

        for i, val in enumerate(display_vals):
            txt = styled_text(f"{val:.2f}", font_size=20, color=TEXT_C)
            entries.add(txt)

        dots = MathTex(r"\vdots", font_size=32, color=DIM)
        entries.add(dots)

        last_vals = list(flat[-2:])
        for val in last_vals:
            txt = styled_text(f"{val:.2f}", font_size=20, color=TEXT_C)
            entries.add(txt)

        entries.arrange(DOWN, buff=0.12)
        entries.move_to(RIGHT * 3.5)

        # Brackets
        brace_l = MathTex(r"\Bigg[", font_size=60, color=BLUE).next_to(entries, LEFT, buff=0.1)
        brace_r = MathTex(r"\Bigg]", font_size=60, color=BLUE).next_to(entries, RIGHT, buff=0.1)

        vec_group = VGroup(brace_l, entries, brace_r)

        vec_label = MathTex(r"\mathbf{x} \in \mathbb{R}^{d}", font_size=30, color=BLUE)
        vec_label.next_to(vec_group, DOWN, buff=0.3)

        self.play(FadeIn(vec_group, shift=RIGHT * 0.3), FadeIn(vec_label), run_time=1.2)
        self.wait(0.8)

        # ── Explanatory text ───────────────────────────────────────────
        explain = styled_text(
            "Each 64×64 face becomes a vector in ℝ⁴⁰⁹⁶",
            font_size=26, color=YELLOW,
        )
        explain.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explain, shift=UP * 0.2))
        self.wait(1.5)

        # ── Animate rows peeling off grid into vector ──────────────────
        # Highlight first row of grid
        first_row = VGroup(*[grid[i] for i in range(8)])
        highlight = SurroundingRectangle(first_row, color=PEACH, buff=0.04)
        self.play(Create(highlight), run_time=0.5)
        self.wait(0.3)

        row_label = styled_text("Row 1 → first 8 entries", font_size=20, color=PEACH)
        row_label.next_to(highlight, LEFT, buff=0.2).shift(DOWN * 0.3)
        self.play(FadeIn(row_label))
        self.wait(1.0)

        self.play(FadeOut(highlight), FadeOut(row_label))
        self.wait(0.5)

        # ── Final emphasis ─────────────────────────────────────────────
        box = SurroundingRectangle(
            VGroup(grid, arrow, vec_group), color=GREEN, buff=0.3, corner_radius=0.1,
        )
        self.play(Create(box))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
