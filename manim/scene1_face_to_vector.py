"""
FaceToVectorScene  —  HIGH QUALITY
Run:  manim -pqh scene1_face_to_vector.py FaceToVectorScene
"""

from manim import *
import numpy as np
import os

config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

BG       = "#252B45"
TEAL     = "#7BAFD4"
ORANGE   = "#F5A623"
OFFWHITE = "#F0EDE6"
DIM      = "#8B9AAF"
DARK     = "#1E2440"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

N    = 64
CELL = 0.090
GAP  = 0.018
STEP = CELL + GAP


def load_faces(n=5):
    xt = os.path.join(DATA_DIR, "X_test.npy")
    if os.path.exists(xt):
        data = np.load(xt)
        return [data[i * 4].reshape(64, 64) for i in range(n)]
    faces = []
    for seed in range(n):
        rng  = np.random.RandomState(seed)
        y, x = np.mgrid[0:64, 0:64]
        f    = np.exp(-((x-32)**2+(y-28)**2)/380)*0.7 + 0.15
        f   += rng.randn(64,64)*0.04
        faces.append(np.clip(f, 0, 1))
    return faces


def gray_hex(v):
    v = int(np.clip(v, 0, 1) * 255)
    return "#{:02X}{:02X}{:02X}".format(v, v, v)


def _sub(n):
    d = {"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄",
         "5":"₅","6":"₆","7":"₇","8":"₈","9":"₉"}
    return "".join(d[c] for c in str(n))


def build_pixel_grid(face, cell=CELL, gap=GAP, center=ORIGIN):
    step = cell + gap
    gw   = N * step - gap
    gh   = N * step - gap
    g    = VGroup()
    cx, cy = center[0], center[1]
    for r in range(N):
        for c in range(N):
            sq = Square(
                side_length  = cell,
                fill_color   = gray_hex(face[r, c]),
                fill_opacity = 1,
                stroke_width = 0,
            )
            sq.move_to([cx - gw/2 + c*step + cell/2,
                        cy + gh/2 - r*step - cell/2, 0])
            g.add(sq)
    return g, gw, gh


class FaceToVectorScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        faces = load_faces(5)
        face  = faces[0]
        vals  = face.flatten()

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1
        # ══════════════════════════════════════════════════════════════════

        title = Text(
            "From Image to Vector",
            font="Calibri", color=OFFWHITE, font_size=52, weight=BOLD,
        )
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN*0.15))

        # ── pixel grid — slightly smaller, left side ──────────────────────
        GCX, GCY = -3.8, 0.0
        # scale down by reducing cell size for this grid only
        SMALL_CELL = 0.075
        SMALL_GAP  = 0.014
        SMALL_STEP = SMALL_CELL + SMALL_GAP
        grid, gw, gh = build_pixel_grid(
            face, cell=SMALL_CELL, gap=SMALL_GAP, center=[GCX, GCY, 0]
        )

        grid_lbl = Text(
            "64 × 64 grayscale image",
            font="Calibri", color=TEAL, font_size=30, weight=BOLD,
        )
        grid_lbl.move_to([GCX, GCY - gh/2 - 0.38, 0])

        self.play(FadeIn(grid), run_time=0.9)
        self.play(FadeIn(grid_lbl))
        self.wait(0.5)

        # ── flatten arrow ─────────────────────────────────────────────────
        ARR_X0 = GCX + gw/2 + 0.20
        ARR_X1 = ARR_X0 + 1.20

        arr = Arrow(
            start=[ARR_X0, GCY, 0], end=[ARR_X1, GCY, 0],
            color=ORANGE, buff=0, stroke_width=8,
            max_tip_length_to_length_ratio=0.22,
        )
        arr_lbl = Text("flatten", font="Calibri",
                       color=ORANGE, font_size=30, weight=BOLD)
        arr_lbl.next_to(arr, UP, buff=0.14)

        self.play(GrowArrow(arr), FadeIn(arr_lbl))
        self.wait(0.3)

        # ── column vector — centred in right half ─────────────────────────
        # Right half: from ARR_X1 to +7.1 (16:9 frame half-width ≈ 7.11)
        RIGHT_CX = (ARR_X1 + 6.9) / 2   # horizontal centre of right half

        # Cell geometry — tall enough to fit value text
        VCH    = 0.55
        VCW    = 1.55
        ROW_SP = VCH * 1.25   # tight spacing so vector fits vertically

        # We show 6 pixel cells + 1 dots row = 7 visible rows
        # Rows: x1, x2, x3, x4, ⋮, x4096
        PIXEL_ROWS = [
            (0,    False),   # (pixel_index, is_dots)
            (1,    False),
            (200,  False),
            (1000, False),
            (-1,   True),    # dots
            (4095, False),
        ]
        N_ROWS     = len(PIXEL_ROWS)
        VEC_HEIGHT = (N_ROWS - 1) * ROW_SP + VCH

        # Centre vector vertically around GCY
        V_TOP = GCY + VEC_HEIGHT / 2

        # -- build cells --
        cell_list   = []   # (rect, pixel_idx) for non-dots rows
        dots_mob    = None
        vec_cells   = VGroup()

        for ri, (idx, is_dots) in enumerate(PIXEL_ROWS):
            cy = V_TOP - ri * ROW_SP - VCH / 2
            if is_dots:
                dots_mob = Text("⋮", font="Calibri",
                                color=DIM, font_size=42)
                dots_mob.move_to([RIGHT_CX, cy, 0])
                vec_cells.add(dots_mob)
                continue
            rect = Rectangle(
                width=VCW, height=VCH,
                fill_color=gray_hex(vals[idx]),
                fill_opacity=1,
                stroke_color=TEAL, stroke_width=1.8,
            )
            rect.move_to([RIGHT_CX, cy, 0])
            cell_list.append((rect, idx))
            vec_cells.add(rect)

        # -- brackets that exactly span the full vector height --
        top_y = V_TOP - VCH / 2               # top edge of first cell
        bot_y = V_TOP - (N_ROWS-1)*ROW_SP - VCH/2  # bottom edge of last cell
        bracket_span_h = top_y - bot_y
        bracket_mid_y  = (top_y + bot_y) / 2

        # Scale brackets to exactly match vector height.
        # font_size 1 ≈ 0.035 Manim units of bracket height (empirical)
        bracket_fs = int(bracket_span_h / 0.033)
        bracket_fs = max(60, min(bracket_fs, 300))

        bL = Text("[", font="Calibri", color=OFFWHITE,
                  font_size=bracket_fs)
        bR = Text("]", font="Calibri", color=OFFWHITE,
                  font_size=bracket_fs)
        bL.move_to([RIGHT_CX - VCW/2 - 0.32, bracket_mid_y, 0])
        bR.move_to([RIGHT_CX + VCW/2 + 0.32, bracket_mid_y, 0])

        # -- x₁ label ABOVE first cell, x₄₀₉₆ BELOW last cell,
        #    "···" on the side between them --
        first_rect = cell_list[0][0]
        last_rect  = cell_list[-1][0]

        lbl_x1 = Text("x₁", font="Calibri", color=DIM, font_size=30)
        lbl_x1.next_to(first_rect, UP, buff=0.14)

        lbl_x4096 = Text("x₄₀₉₆", font="Calibri", color=DIM, font_size=30)
        lbl_x4096.next_to(last_rect, DOWN, buff=0.14)

        # side dots to the right of brackets
        side_dots = Text("···", font="Calibri", color=DIM, font_size=28)
        side_dots.next_to(bR, RIGHT, buff=0.25)

        # -- pixel value texts (initially invisible, revealed later) --
        val_texts = VGroup()
        for rect, idx in cell_list:
            vt = Text(f"{vals[idx]:.2f}", font="Calibri",
                      color=OFFWHITE, font_size=22, weight=BOLD)
            vt.move_to(rect.get_center())
            val_texts.add(vt)

        # -- animate in --
        self.play(
            FadeIn(vec_cells, lag_ratio=0.10, run_time=0.9),
            FadeIn(bL), FadeIn(bR),
        )
        self.play(FadeIn(lbl_x1), FadeIn(lbl_x4096), FadeIn(side_dots))
        self.wait(0.5)

        # -- reveal pixel values inside cells --
        # "pixel values ∈ [0,1]" note goes at the BOTTOM of the screen
        val_note = Text("pixel values  ∈  [0, 1]",
                        font="Calibri", color=ORANGE, font_size=30, weight=BOLD)
        val_note.move_to([RIGHT_CX, lbl_x4096.get_bottom()[1] - 0.42, 0])

        self.play(FadeIn(val_note, shift=UP*0.15))
        self.wait(0.15)
        self.play(
            LaggedStart(*[FadeIn(vt) for vt in val_texts],
                        lag_ratio=0.12, run_time=0.9)
        )
        self.wait(1.3)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3 — 5 images + column vectors
        # ══════════════════════════════════════════════════════════════════

        self.play(
            FadeOut(Group(
                title, grid, grid_lbl, arr, arr_lbl,
                vec_cells, bL, bR, lbl_x1, lbl_x4096,
                side_dots, val_note, val_texts,
            )),
            run_time=0.5,
        )

        title2 = Text(
            "Each Image Becomes a Column Vector",
            font="Calibri", color=OFFWHITE, font_size=48, weight=BOLD,
        )
        title2.to_edge(UP, buff=0.35)
        self.play(FadeIn(title2))

        N_FACES = 5
        IMG_H   = 1.55
        IMG_GAP = 0.38
        TOT_W   = N_FACES*IMG_H + (N_FACES-1)*IMG_GAP
        IMG_X0  = -TOT_W/2 + IMG_H/2
        IMG_Y   = 1.90

        mini_imgs = Group()
        img_lbls  = VGroup()
        img_xs    = []

        for i, f in enumerate(faces):
            np_arr = (f[:,:,None]*np.ones((1,1,3))*255).astype(np.uint8)
            im = ImageMobject(np_arr)
            im.set_height(IMG_H)
            cx = IMG_X0 + i*(IMG_H + IMG_GAP)
            img_xs.append(cx)
            im.move_to([cx, IMG_Y, 0])
            mini_imgs.add(im)
            lbl = Text(f"image {i+1}", font="Calibri",
                       color=DIM, font_size=24)
            lbl.next_to(im, UP, buff=0.12)
            img_lbls.add(lbl)

        self.play(
            LaggedStart(*[FadeIn(m) for m in mini_imgs],
                        lag_ratio=0.12, run_time=1.0),
            FadeIn(img_lbls),
        )
        self.wait(0.35)

        # flatten arrows
        dn_arrows = VGroup()
        for cx in img_xs:
            bot_y = IMG_Y - IMG_H/2
            a = Arrow(
                start=[cx, bot_y - 0.06, 0],
                end  =[cx, bot_y - 0.62, 0],
                color=ORANGE, buff=0, stroke_width=5,
                max_tip_length_to_length_ratio=0.28,
            )
            dn_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in dn_arrows],
                              lag_ratio=0.10, run_time=0.9))

        # column vectors
        CV_CH   = 0.30
        CV_W    = IMG_H * 0.65
        CV_RSP  = CV_CH * 1.35
        CV_TOP_Y= IMG_Y - IMG_H/2 - 0.78

        CV_ROWS = [(0, False), (-1, True), (4095, False)]
        N_CV    = len(CV_ROWS)
        cv_h    = (N_CV-1)*CV_RSP + CV_CH
        cv_mid_y= CV_TOP_Y - cv_h/2 + CV_CH*0.25

        col_vec_grps = []
        col_vec_bLs  = []
        col_vec_bRs  = []
        col_vec_lbls = VGroup()

        for i, f in enumerate(faces):
            cx  = img_xs[i]
            fv  = f.flatten()
            grp = VGroup()

            for ri, (idx, is_dots) in enumerate(CV_ROWS):
                cy = CV_TOP_Y - ri*CV_RSP - CV_CH/2
                if is_dots:
                    d = Text("⋮", font="Calibri", color=DIM, font_size=20)
                    d.move_to([cx, cy, 0])
                    grp.add(d)
                    continue
                rect = Rectangle(
                    width=CV_W, height=CV_CH,
                    fill_color=gray_hex(fv[idx]),
                    fill_opacity=1,
                    stroke_color=TEAL, stroke_width=1.2,
                )
                rect.move_to([cx, cy, 0])
                grp.add(rect)

            # brackets sized to vector
            cv_top_edge = CV_TOP_Y - CV_CH/2
            cv_bot_edge = CV_TOP_Y - (N_CV-1)*CV_RSP - CV_CH/2
            cv_span     = cv_top_edge - cv_bot_edge
            cv_bmid     = (cv_top_edge + cv_bot_edge) / 2
            cv_bfs      = int(cv_span / 0.033)
            cv_bfs      = max(40, min(cv_bfs, 200))

            bl = Text("[", font="Calibri", color=OFFWHITE, font_size=cv_bfs)
            br = Text("]", font="Calibri", color=OFFWHITE, font_size=cv_bfs)
            bl.move_to([cx - CV_W/2 - 0.18, cv_bmid, 0])
            br.move_to([cx + CV_W/2 + 0.18, cv_bmid, 0])

            col_vec_grps.append(grp)
            col_vec_bLs.append(bl)
            col_vec_bRs.append(br)

            xl = Text(f"x{_sub(i+1)}", font="Calibri",
                      color=DIM, font_size=26)
            xl.move_to([cx, CV_TOP_Y - cv_h - 0.32, 0])
            col_vec_lbls.add(xl)

        self.play(
            LaggedStart(*[FadeIn(it) for it in
                         col_vec_grps + col_vec_bLs + col_vec_bRs],
                        lag_ratio=0.08, run_time=1.2),
            FadeIn(col_vec_lbls),
        )
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4 — vectors only slide together; images stay fixed
        # ══════════════════════════════════════════════════════════════════

        self.play(FadeOut(dn_arrows), FadeOut(img_lbls),
                  FadeOut(col_vec_lbls))

        MAT_COL_W = CV_W + 0.05
        MAT_TOTAL = N_FACES*MAT_COL_W + (N_FACES-1)*0.05
        MAT_CX    = 0.0
        MAT_X0    = MAT_CX - MAT_TOTAL/2 + MAT_COL_W/2
        tgt_xs    = [MAT_X0 + i*(MAT_COL_W+0.05) for i in range(N_FACES)]

        vec_slide = [
            grp.animate.move_to([tgt_xs[i], cv_mid_y, 0])
            for i, grp in enumerate(col_vec_grps)
        ]
        bracket_fade = [FadeOut(b) for b in col_vec_bLs + col_vec_bRs]

        self.play(
            LaggedStart(*vec_slide, lag_ratio=0.08, run_time=1.4),
            *bracket_fade,
        )
        self.wait(0.3)

        # matrix brackets
        mat_bL = Text("[", font="Calibri", color=ORANGE, font_size=160)
        mat_bR = Text("]", font="Calibri", color=ORANGE, font_size=160)
        mat_bL.move_to([MAT_CX - MAT_TOTAL/2 - 0.45, cv_mid_y, 0])
        mat_bR.move_to([MAT_CX + MAT_TOTAL/2 + 0.45, cv_mid_y, 0])

        self.play(FadeIn(mat_bL), FadeIn(mat_bR))

        # ── X ∈ ℝ⁴⁰⁹⁶ˣᴺ  label — single line, directly below the matrix ──
        mat_lbl = Text(
            "X  ∈  ℝ⁴⁰⁹⁶ˣᴺ     —     each column = one flattened face image",
            font="Calibri", color=OFFWHITE, font_size=26, weight=BOLD,
        )
        mat_lbl.next_to(mat_bR.get_bottom(), DOWN, buff=0.35)
        mat_lbl.set_x(MAT_CX)   # horizontally centred under the matrix

        self.play(Write(mat_lbl))

        # d=4096 double arrow on left
        dim_arrow = DoubleArrow(
            start=[MAT_CX - MAT_TOTAL/2 - 0.85, cv_mid_y - cv_h/2, 0],
            end  =[MAT_CX - MAT_TOTAL/2 - 0.85, cv_mid_y + cv_h/2, 0],
            color=DIM, buff=0, stroke_width=3, tip_length=0.15,
        )
        dim_lbl = Text("d = 4096", font="Calibri", color=DIM, font_size=26)
        dim_lbl.next_to(dim_arrow, LEFT, buff=0.14)

        self.play(Create(dim_arrow), FadeIn(dim_lbl))
        self.wait(3.0)
