"""
helpers.py — Shared utilities for all Manim scenes.

Provides a consistent visual language (colours, fonts, spacing) and
reusable helper functions so that every scene looks like part of the
same presentation deck.
"""

from manim import *
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Colour palette  (harmonious, accessible, slide-friendly)
# ──────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":          "#1e1e2e",     # dark background
    "text":        "#cdd6f4",     # light text
    "accent1":     "#89b4fa",     # blue
    "accent2":     "#f38ba8",     # pink / red
    "accent3":     "#a6e3a1",     # green
    "accent4":     "#fab387",     # peach / orange
    "accent5":     "#cba6f7",     # lavender
    "accent6":     "#f9e2af",     # yellow
    "dim":         "#585b70",     # subtle grey
    "white":       "#ffffff",
}

# Quick MObject colour accessors
BG      = PALETTE["bg"]
TEXT_C  = PALETTE["text"]
BLUE    = PALETTE["accent1"]
PINK    = PALETTE["accent2"]
GREEN   = PALETTE["accent3"]
PEACH   = PALETTE["accent4"]
LAVEN   = PALETTE["accent5"]
YELLOW  = PALETTE["accent6"]
DIM     = PALETTE["dim"]

# Class colours for scatter plots (10 distinct)
CLASS_COLORS = [
    "#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7",
    "#f9e2af", "#94e2d5", "#f5c2e7", "#74c7ec", "#eba0ac",
]


# ──────────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────────

def styled_text(content, font_size=32, color=TEXT_C, weight=NORMAL, **kw):
    """Create a consistently styled Text mobject."""
    return Text(content, font_size=font_size, color=color, weight=weight, **kw)


def section_title(content, font_size=42):
    """Bold section title at top of frame."""
    return styled_text(content, font_size=font_size, weight=BOLD).to_edge(UP, buff=0.5)


def math(tex_string, font_size=38, color=TEXT_C):
    """Shorthand for MathTex with project styling."""
    return MathTex(tex_string, font_size=font_size, color=color)


# ──────────────────────────────────────────────────────────────────────────
# Pixel-grid face helper
# ──────────────────────────────────────────────────────────────────────────

def create_pixel_grid(image_array, grid_size=8, cell_size=0.35, stroke_width=0.5):
    """
    Build a Manim VGroup grid from a 2-D numpy image array.

    Parameters
    ----------
    image_array : ndarray, shape (H, W)  — values in [0, 1].
    grid_size   : int — resample to grid_size × grid_size.
    cell_size   : float — side length of each square in scene units.
    stroke_width : float

    Returns
    -------
    grid : VGroup of Squares
    """
    from PIL import Image as PILImage

    # Resample to target grid resolution
    if image_array.shape != (grid_size, grid_size):
        pil = PILImage.fromarray((image_array * 255).astype(np.uint8), mode="L")
        pil = pil.resize((grid_size, grid_size), PILImage.BILINEAR)
        image_array = np.array(pil).astype(float) / 255.0

    grid = VGroup()
    for r in range(grid_size):
        for c in range(grid_size):
            val = float(image_array[r, c])
            sq = Square(side_length=cell_size)
            sq.set_fill(color=rgb_to_color([val, val, val]), opacity=1)
            sq.set_stroke(WHITE, width=stroke_width)
            sq.move_to(np.array([
                c * cell_size - (grid_size - 1) * cell_size / 2,
                -(r * cell_size - (grid_size - 1) * cell_size / 2),
                0,
            ]))
            grid.add(sq)
    return grid


def create_face_thumbnail(image_array, height=2.0):
    """
    Create a Manim ImageMobject from a numpy array.

    Parameters
    ----------
    image_array : ndarray, shape (H, W) — values in [0, 1].
    height : float — desired height in scene units.
    """
    uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    # Stack to 3-channel for ImageMobject
    rgb = np.stack([uint8, uint8, uint8], axis=-1)
    img = ImageMobject(rgb)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = height
    return img


# ──────────────────────────────────────────────────────────────────────────
# Transition helpers
# ──────────────────────────────────────────────────────────────────────────

def fade_replace(scene, old_group, new_group, run_time=0.8):
    """Fade out old mobjects and fade in new ones."""
    scene.play(FadeOut(old_group), run_time=run_time / 2)
    scene.play(FadeIn(new_group), run_time=run_time / 2)
