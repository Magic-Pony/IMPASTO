"""Independent contact-driven pigment model, not the planner's prerenderer.

Approximates footprint spreading, film coverage, pigment load and deterministic
canvas tooth. It is NOT a Navier-Stokes / bristle-resolved oil paint simulator.
"""
from dataclasses import dataclass
import numpy as np

PALETTE = np.array([
    [35,94,162], [217,61,54], [236,180,40], [51,130,84],
    [247,246,235], [39,34,35], [133,78,46], [212,128,66],
    [97,159,193], [234,126,148], [118,88,164], [44,145,151],
    [112,155,69], [42,82,61], [147,65,67], [204,179,133],
    [93,117,137], [58,73,113], [235,219,164], [187,116,69],
    [151,115,146], [103,72,51], [238,160,93], [151,190,190],
], dtype=float) / 255


def footprint_weight(normalized_distance_sq, config):
    """Evaluate the configured radial pigment-deposition profile."""
    d2 = np.asarray(normalized_distance_sq, dtype=np.float32)
    profile = config.get("footprint_profile", "linear")
    if profile == "linear":
        return np.clip((1.0 - d2) * 3.0, 0.0, 1.0)
    if profile != "gaussian":
        raise ValueError(f"Unknown paint footprint profile: {profile}")

    radius = np.sqrt(np.maximum(d2, 0.0))
    sigma = float(config.get("footprint_sigma_fraction", 0.72))
    taper_start = float(config.get("footprint_taper_start_fraction", 0.75))
    if sigma <= 0.0 or not 0.0 <= taper_start < 1.0:
        raise ValueError("Invalid Gaussian footprint sigma or taper start")
    gaussian = np.exp(-0.5 * d2 / (sigma * sigma))
    t = np.clip((radius - taper_start) / (1.0 - taper_start), 0.0, 1.0)
    smoothstep = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    taper = np.where(radius <= taper_start, 1.0, 1.0 - smoothstep)
    return np.clip(np.where(d2 < 1.0, gaussian * taper, 0.0), 0.0, 1.0)


@dataclass
class BrushState:
    color: int = -1
    load: float = 0.0
    opacity: float = 1.0
    clean_cycles: int = 0

    def clean(self):
        self.color = -1
        self.load = 0.0
        self.clean_cycles += 1

    def dip(self, color, opacity=1.0):
        self.color = int(color)
        self.opacity = float(opacity)
        self.load = 1.0


class Canvas:
    def __init__(self, width=.4572, height=.5334, resolution=512, seed=0,
                 paint_config=None):
        self.width, self.height, self.resolution = width, height, resolution
        self.paint_config = dict(paint_config or {})
        self.palette = np.asarray(self.paint_config.get("palette_rgb", PALETTE), dtype=float)
        if self.palette.max() > 1:
            self.palette = self.palette / 255.0
        rng = np.random.default_rng(seed)
        self.tooth = rng.uniform(.85, 1.15, (resolution, resolution)).astype(np.float32)
        self.rgb = np.ones((resolution, resolution, 3), np.float32) * np.array([.963,.952,.918])
        self.initial = self.rgb.copy()
        self.film = np.zeros((resolution, resolution), np.float32)
        self.last = None
        self.deposited_distance = 0.0

    def pixel(self, xy):
        return np.array([(xy[0]/self.width+.5)*(self.resolution-1),
                         (.5-xy[1]/self.height)*(self.resolution-1)])

    def reset_path(self):
        self.last = None

    def update(self, xy, force, brush, dt, contacting=True):
        xy = np.asarray(xy, dtype=float)
        if not contacting or force <= .02 or brush.load <= 1e-4 or brush.color < 0:
            self.last = None
            return
        prev = xy if self.last is None else self.last
        distance = float(np.linalg.norm(xy-prev))
        # One traveled distance determines deposition irrespective of substeps.
        n = max(1, int(np.ceil(distance/.0006)))
        radius_cfg = self.paint_config.get("force_radius_m", {})
        radius = (radius_cfg.get("minimum", .0018)
                  + radius_cfg.get("scale", .0042)
                  * max(force, 0)**radius_cfg.get("exponent", .55))
        for k in range(1, n+1):
            pos = prev+(xy-prev)*k/n
            self._stamp(pos, radius, (distance/n+.002*dt/n), brush)
        brush.load = max(0., brush.load-distance*.32-dt*.0005)
        self.deposited_distance += distance
        self.last = xy.copy()

    def _stamp(self, xy, radius, ds, brush):
        n = self.resolution
        px, py = self.pixel(xy)
        rx, ry = radius/self.width*(n-1), radius/self.height*(n-1)
        x0, x1 = max(0, int(px-rx-2)), min(n, int(px+rx+3))
        y0, y1 = max(0, int(py-ry-2)), min(n, int(py+ry+3))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = ((xx-px)/rx)**2 + ((yy-py)/ry)**2
        footprint = footprint_weight(d, self.paint_config)
        coverage = 1-np.exp(-ds/.0008 * footprint * self.tooth[y0:y1,x0:x1]
                           * brush.load * brush.opacity)
        region = self.rgb[y0:y1,x0:x1]
        color = self.palette[brush.color]
        region[:] = region*(1-coverage[...,None])+color*coverage[...,None]
        self.film[y0:y1,x0:x1] += (coverage*.015).astype(np.float32)

    def image(self):
        return np.uint8(np.clip(self.rgb*255, 0, 255))

    def changed_mask(self):
        return np.max(np.abs(self.rgb-self.initial),axis=-1) > .04
