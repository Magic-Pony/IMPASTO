"""Reproducible Stage-3 targets extracted from the authors' public Fig. 8.

The paper does not release the original high-resolution artist files or the
per-stroke action labels.  The public planning figure does contain the target
panels and intermediate checkpoints, so those panels are used here as the
image goals while the action sequence remains inferred by the planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from impasto.actions import Stroke
from impasto.sim.paint import BrushState, Canvas
from impasto.stage2.raster import srgb_gray


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_PLANNING_FIGURE = ROOT / "papers/official_media/planning-2.png"

# Coordinates in the authors' public planning figure. The first seven cells
# are Flower; the second seven are Fish. They show exactly the target frames
# used by Fig. 8, including the step-5/10/15/final MPC windows.
_OFFICIAL_X = {
    "flower": ([(0, 216), (217, 410), (411, 630), (631, 824),
                 (825, 1047), (1048, 1241), (1253, 1448)]),
    "fish": ([(1453, 1653), (1665, 1860), (1866, 2070), (2079, 2274),
               (2280, 2487), (2497, 2692), (2703, 2898)]),
}
_OFFICIAL_COUNTS = (2, 5, 7, 10, 12, 15, None)


@dataclass(frozen=True)
class TargetArtwork:
    name: str
    strokes: tuple[Stroke, ...]
    image_rgb: np.ndarray
    image_gray: np.ndarray
    source: str = "authors' public Fig. 8 target crop"


def _flower_strokes() -> list[Stroke]:
    # All actions stay within the Stage-2 verified sampler bounds.
    return [
        Stroke(-.055, -.105, .082, .010, 1.48, .72, 12),
        Stroke(-.047, -.061, .052, .012, .28, .66, 12),
        Stroke(-.050, -.062, .051, -.012, 2.82, .63, 12),
        Stroke(-.031, -.035, .048, -.009, .70, .64, 12),
        Stroke(-.033, -.035, .050, .010, 2.42, .64, 12),
        Stroke(-.035, .011, .058, .010, .12, .78, 1),
        Stroke(-.035, .011, .058, -.010, .12, .74, 1),
        Stroke(-.035, .011, .056, .009, 1.26, .72, 1),
        Stroke(-.035, .011, .056, -.009, 1.26, .72, 1),
        Stroke(-.035, .011, .055, .010, 2.38, .74, 1),
        Stroke(-.035, .011, .055, -.010, 2.38, .74, 1),
        Stroke(-.035, .011, .050, .008, -1.02, .68, 1),
        Stroke(-.035, .011, .050, -.008, -1.02, .68, 1),
        Stroke(-.035, .011, .024, .003, .35, .90, 7),
        Stroke(-.035, .011, .024, -.003, 1.35, .90, 7),
        Stroke(-.035, .011, .024, .003, 2.35, .90, 7),
        Stroke(-.035, .011, .024, -.003, -2.65, .90, 7),
    ]


def _fish_strokes() -> list[Stroke]:
    return [
        Stroke(-.105, .018, .072, .012, .02, .62, 0),
        Stroke(-.039, .020, .071, -.012, .02, .62, 0),
        Stroke(-.105, .018, .070, -.012, .02, .60, 0),
        Stroke(-.039, .020, .069, .012, .02, .60, 0),
        Stroke(-.070, .020, .054, .010, .02, .58, 0),
        Stroke(-.070, .020, .054, -.010, .02, .58, 0),
        Stroke(-.038, .020, .044, .016, 2.22, .66, 0),
        Stroke(-.040, .020, .044, -.016, -2.22, .66, 0),
        Stroke(-.020, .020, .036, .010, 1.57, .67, 0),
        Stroke(-.020, .020, .036, -.010, -1.57, .67, 0),
        Stroke(-.087, .020, .022, .004, .05, .82, 7),
        Stroke(-.079, .020, .018, -.003, .05, .78, 7),
        Stroke(-.057, .041, .037, .008, 1.20, .70, 1),
        Stroke(-.057, -.002, .037, -.008, -1.20, .70, 1),
        Stroke(-.005, .056, .022, .004, 1.57, .75, 12),
        Stroke(.005, .079, .018, -.003, 1.57, .72, 12),
        Stroke(.015, .100, .014, .002, 1.57, .70, 12),
        Stroke(-.010, .096, .012, .002, 1.57, .68, 12),
    ]


def artwork_strokes(name: str) -> tuple[Stroke, ...]:
    key = name.lower()
    if key == "flower":
        strokes = _flower_strokes()
    elif key == "fish":
        strokes = _fish_strokes()
    else:
        raise ValueError(f"Unknown Stage-3 artwork: {name}")
    return tuple(strokes)


def _official_target(name, stroke_count, resolution, seed):
    if not OFFICIAL_PLANNING_FIGURE.is_file():
        return None
    key = name.lower()
    if key not in _OFFICIAL_X:
        return None
    count = None if stroke_count is None else int(stroke_count)
    if count not in (*_OFFICIAL_COUNTS, len(artwork_strokes(key))):
        return None
    index = {2: 0, 5: 1, 7: 2, 10: 3, 12: 4, 15: 5,
             None: 6, len(artwork_strokes(key)): 6}[count]
    x0, x1 = _OFFICIAL_X[key][index]
    # Remove the step title and the outer frame, retaining the complete target
    # image cell. The figure is low resolution, so preserve its aspect ratio
    # and letterbox it instead of stretching brush geometry.
    source = Image.open(OFFICIAL_PLANNING_FIGURE).convert("RGB")
    if key == "flower" and index == 0:
        # Figure row label, outside the painting. Never treat it as pigment.
        source = source.copy()
        source.paste((237, 238, 237), (4, 203, 140, 239))
    # The source figure's frame has a two-pixel antialiased halo below the
    # nominal top edge. Start well inside the panel and stop before the lower
    # separator so that the horizontal frame is never presented as paint.
    tile = source.crop((x0 + 4, 55, x1 - 4, 240))
    # Place the entire photographed panel inside the verified robot workspace;
    # a metric square is nonsquare in the square canvas texture (18 x 21 in).
    tile = tile.resize((round(resolution * .285 / .4572),
                        round(resolution * .285 / .5334)), Image.Resampling.LANCZOS)
    background = np.array([.963, .952, .918], dtype=np.float32) * 255.0
    image = np.broadcast_to(background.astype(np.uint8), (resolution, resolution, 3)).copy()
    pixels = np.asarray(tile, dtype=np.float32)
    # Match the panel's near-white background to the actual stage-1 canvas,
    # while retaining antialiased colored brush edges.
    corners = np.concatenate([pixels[:8, :8].reshape(-1, 3),
                              pixels[:8, -8:].reshape(-1, 3),
                              pixels[-8:, :8].reshape(-1, 3),
                              pixels[-8:, -8:].reshape(-1, 3)])
    panel_bg = np.median(corners, axis=0)
    delta = (pixels - panel_bg) / 255.
    chroma = np.ptp(delta, axis=-1)
    dark = np.mean(-delta, axis=-1)
    alpha = np.maximum(np.clip((chroma - .035) / .025, 0, 1),
                       np.clip((dark - .11) / .05, 0, 1))[..., None]
    pixels = pixels / panel_bg * background
    pixels = background[None, None] * (1.0 - alpha) + pixels * alpha
    y0 = (resolution - pixels.shape[0]) // 2
    x0 = (resolution - pixels.shape[1]) // 2 - round(.015 / .4572 * resolution)
    image[y0:y0 + pixels.shape[0], x0:x0 + pixels.shape[1]] = np.uint8(np.clip(pixels, 0, 255))
    strokes = artwork_strokes(key)
    return TargetArtwork(key, strokes[:count] if count is not None else strokes,
                         image, srgb_gray(image), source="authors' public Fig. 8 target crop")


def render_target_prefix(name: str, stroke_count=None, width=.4572, height=.5334,
                         resolution=512, seed=0, paint_config=None) -> TargetArtwork:
    official = _official_target(name, stroke_count, resolution, seed)
    if official is not None:
        return official
    canvas = Canvas(width, height, resolution, seed=seed, paint_config=paint_config)
    brush = BrushState()
    all_strokes = artwork_strokes(name)
    strokes = all_strokes[:stroke_count] if stroke_count is not None else all_strokes
    for stroke in strokes:
        if brush.color != stroke.color or brush.load < .08:
            brush.dip(stroke.color, stroke.opacity)
        canvas.reset_path()
        # The fast canvas is the same contact-driven deposition law used by
        # Stage 1, with a small integration step for a stable target image.
        for s in np.linspace(0.0, 1.0, 160):
            canvas.update(stroke.points(s), stroke.force, brush, .004, contacting=True)
    rgb = canvas.image()
    return TargetArtwork(name, tuple(strokes), rgb, srgb_gray(rgb), source=(
        "local deterministic surrogate; official target crop unavailable"))


def render_target(name: str, width=.4572, height=.5334, resolution=512,
                  seed=0, paint_config=None) -> TargetArtwork:
    return render_target_prefix(name, None, width, height, resolution, seed, paint_config)


def render_single_target(stroke: Stroke, width=.4572, height=.5334, resolution=512,
                         seed=0, paint_config=None) -> TargetArtwork:
    canvas = Canvas(width, height, resolution, seed=seed, paint_config=paint_config)
    brush = BrushState()
    brush.dip(stroke.color, stroke.opacity)
    canvas.reset_path()
    for s in np.linspace(0.0, 1.0, 160):
        canvas.update(stroke.points(s), stroke.force, brush, .004, contacting=True)
    rgb = canvas.image()
    return TargetArtwork("single_stroke", (stroke,), rgb, srgb_gray(rgb), source="local generated single stroke")
