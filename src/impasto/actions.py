"""Paper-compatible planar quadratic Bezier action, SI units."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Stroke:
    x: float
    y: float
    length: float
    bend: float
    angle: float
    force: float
    color: int = 0
    opacity: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.vector()).all():
            raise ValueError("Stroke parameters must be finite")
        if self.length <= 0 or not 0 <= self.force <= 4:
            raise ValueError("Positive length and force in [0,4] N required")
        if not 0 <= self.color < 24 or not 0 <= self.opacity <= 1:
            raise ValueError("Invalid pigment/opacity")

    def vector(self):
        return np.array([self.x, self.y, self.length, self.bend, self.angle, self.force])

    def points(self, s):
        s = np.asarray(s)
        tangent = np.array([np.cos(self.angle), np.sin(self.angle)])
        normal = np.array([-np.sin(self.angle), np.cos(self.angle)])
        return (np.array([self.x, self.y]) + s[..., None] * self.length * tangent
                + (2*s*(1-s))[..., None] * self.bend * normal)

    def arc_length(self):
        return float(np.linalg.norm(np.diff(self.points(np.linspace(0, 1, 101)), axis=0), axis=1).sum())

    def to_dict(self):
        return asdict(self)


def sample_strokes(seed, count=5):
    """Workspace-contained random self-play, independent of worker assignment."""
    rng = np.random.default_rng(seed)
    strokes = []
    for _ in range(count):
        for _attempt in range(1000):
            s = Stroke(float(rng.uniform(-.11, .04)), float(rng.uniform(-.11, .11)),
                       float(rng.uniform(.025, .085)), float(rng.uniform(-.035, .035)),
                       float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(.4, 1.5)),
                       int(rng.integers(0, 24)))
            p = s.points(np.linspace(0, 1, 30))
            if np.abs(p[:, 0]).max() < .15 and np.abs(p[:, 1]).max() < .16:
                strokes.append(s)
                break
        else:
            raise RuntimeError("Could not sample safe stroke")
    return strokes
