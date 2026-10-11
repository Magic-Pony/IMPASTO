"""Paper-compatible local rasterization and crop utilities for Stage 2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from impasto.actions import Stroke
from impasto.sim.paint import PALETTE


@dataclass(frozen=True)
class CropTransform:
    center_x_m: float
    center_y_m: float
    side_m: float
    patch_size: int = 100

    def world_to_patch(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([
            (xy[..., 0] - self.center_x_m) / self.side_m + .5,
            .5 - (xy[..., 1] - self.center_y_m) / self.side_m,
        ], axis=-1) * (self.patch_size - 1)

    def to_dict(self):
        return {"center_x_m": self.center_x_m, "center_y_m": self.center_y_m,
                "side_m": self.side_m, "patch_size": self.patch_size}


def srgb_gray(rgb):
    """Return the paper-style single luminance channel in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.0:
        rgb = rgb / 255.0
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        return np.clip(.299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2], 0, 1)
    return np.asarray(rgb, dtype=np.float32)


def crop_canvas_gray(rgb, crop: CropTransform, canvas_width=.4572, canvas_height=.5334):
    """Crop an RGB canvas in metric coordinates and resize to the model patch."""
    gray = np.uint8(np.clip(srgb_gray(rgb) * 255.0, 0, 255))
    image = Image.fromarray(gray, mode="L")
    width, height = image.size
    cx = (crop.center_x_m / canvas_width + .5) * (width - 1)
    cy = (.5 - crop.center_y_m / canvas_height) * (height - 1)
    half_w = crop.side_m / canvas_width * (width - 1) / 2
    half_h = crop.side_m / canvas_height * (height - 1) / 2
    box = (int(round(cx - half_w)), int(round(cy - half_h)),
           int(round(cx + half_w)), int(round(cy + half_h)))
    # The action sampler keeps this inside the canvas. Padding here makes the
    # contract well-defined for future boundary-focused data collection.
    pad = max(0, -box[0], -box[1], box[2] - width, box[3] - height)
    if pad:
        image = Image.fromarray(np.pad(np.asarray(image), pad, mode="edge"), mode="L")
        box = tuple(v + pad for v in box)
    patch = image.crop(box).resize((crop.patch_size, crop.patch_size), Image.Resampling.BILINEAR)
    return np.asarray(patch, dtype=np.float32) / 255.0


def crop_for_stroke(stroke: Stroke, patch_size=100, margin_m=.025):
    pts = stroke.points(np.linspace(0, 1, 256))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = max(float(hi[0] - lo[0]), float(hi[1] - lo[1])) + 2 * margin_m
    side = max(side, .05)
    return CropTransform(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), side, patch_size)


def force_radius(force, radius_cfg):
    return (float(radius_cfg["minimum"])
            + float(radius_cfg["scale"]) * max(float(force), 0.0) ** float(radius_cfg["exponent"]))


def color_gray(color, palette=None):
    rgb = np.asarray(PALETTE if palette is None else palette)[int(color)].astype(float)
    if rgb.max() > 1:
        rgb /= 255.
    return float(np.dot(rgb, np.array([.299, .587, .114], dtype=np.float32)))


def render_stroke_patch(stroke: Stroke, crop: CropTransform, radius_cfg, color_value=None,
                        samples=None, palette=None):
    """Rasterize a filled disk-stamped quadratic Bezier footprint to [0, 1]."""
    image = np.ones((crop.patch_size, crop.patch_size), dtype=np.float32)
    radius = force_radius(stroke.force, radius_cfg)
    points = stroke.points(np.linspace(0, 1, int(samples or 256)))
    pixels = crop.world_to_patch(points)
    rp = radius / crop.side_m * (crop.patch_size - 1)
    value = color_gray(stroke.color, palette) if color_value is None else float(color_value)
    # The union of equal-radius disk stamps is exactly the set of pixels
    # within rp of the nearest sampled center. A spatial index avoids 256
    # Python stamping loops for every candidate, without changing the
    # centers, radius, antialiasing contract or pigment value.
    lo = np.maximum(0, np.floor(pixels.min(axis=0) - rp).astype(int))
    hi = np.minimum(crop.patch_size, np.ceil(pixels.max(axis=0) + rp).astype(int) + 1)
    x0, y0 = lo
    x1, y1 = hi
    if x1 <= x0 or y1 <= y0:
        return image
    yy, xx = np.mgrid[y0:y1, x0:x1]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    _, nearest = cKDTree(pixels).query(grid)
    distances_sq = np.sum((grid - pixels[nearest]) ** 2, axis=1)
    mask = (distances_sq <= rp ** 2).reshape(y1-y0, x1-x0)
    image[y0:y1, x0:x1][mask] = min(value, 1.0)
    return image


def patch_action(stroke: Stroke, crop: CropTransform):
    """Return local pixel geometry in image coordinates plus force and angle."""
    p0 = crop.world_to_patch([stroke.x, stroke.y])
    alpha_image = -float(stroke.angle)
    bend_image = -float(stroke.bend) / crop.side_m * (crop.patch_size - 1)
    length_pixels = float(stroke.length) / crop.side_m * (crop.patch_size - 1)
    return np.array([p0[0], p0[1], length_pixels, bend_image,
                     float(stroke.force), alpha_image], dtype=np.float32)


def normalize_patch_action(action, force_bounds, patch_size=100, bend_half_range_px=None):
    """Normalize local geometry, force, and angle while preserving bend sign.

    The paper does not state the action normalization. We use [0, 1] inputs for
    the encoder and map the signed pixel bend symmetrically around 0.5. The
    default half range is one patch width, which covers the local sampler.
    """
    action = np.asarray(action, dtype=np.float32).copy()
    scale = float(max(patch_size - 1, 1))
    action[[0, 1, 2, 3]] /= scale
    half_range = float(bend_half_range_px or scale)
    action[3] = .5 + .5 * action[3] * scale / max(half_range, 1e-8)
    action[4] = (action[4] - force_bounds[0]) / max(force_bounds[1] - force_bounds[0], 1e-8)
    action[5] = (action[5] + np.pi) / (2 * np.pi)
    return np.clip(action, 0, 1)
