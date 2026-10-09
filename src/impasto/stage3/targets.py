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
    stroke_count: int = 1


ARTWORK_COUNTS = {"flower": 17, "fish": 18}  # Main paper, Sec. IV Q3.


def _official_target(name, stroke_count, resolution, seed):
    if not OFFICIAL_PLANNING_FIGURE.is_file():
        return None
    key = name.lower()
    if key not in _OFFICIAL_X:
        return None
    count = None if stroke_count is None else int(stroke_count)
    if count not in (*_OFFICIAL_COUNTS, ARTWORK_COUNTS[key]):
        return None
    index = {2: 0, 5: 1, 7: 2, 10: 3, 12: 4, 15: 5,
             None: 6, ARTWORK_COUNTS[key]: 6}[count]
    x0, x1 = _OFFICIAL_X[key][index]
    # Remove the step title and the outer frame, retaining the complete target
    # image cell. Each published panel is normalized to the same metric square.
    # This is a local figure-extraction convention, not a camera calibration.
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
    return TargetArtwork(key, (), image, srgb_gray(image),
                         source="authors' public Fig. 8 target crop",
                         stroke_count=count if count is not None else ARTWORK_COUNTS[key])


def render_target_prefix(name: str, stroke_count=None, width=.4572, height=.5334,
                         resolution=512, seed=0, paint_config=None) -> TargetArtwork:
    official = _official_target(name, stroke_count, resolution, seed)
    if official is not None:
        return official
    raise FileNotFoundError("Official artwork frame unavailable; refusing surrogate fallback")


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
