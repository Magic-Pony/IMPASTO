import numpy as np

from impasto.actions import Stroke


def test_bezier_endpoints_and_midpoint_bend():
    stroke = Stroke(0.1, -0.2, 0.4, 0.08, 0.0, 0.5, color=3)
    np.testing.assert_allclose(stroke.points(0.0), [0.1, -0.2])
    np.testing.assert_allclose(stroke.points(1.0), [0.5, -0.2])
    np.testing.assert_allclose(stroke.points(0.5), [0.3, -0.16])


def test_invalid_force_is_rejected():
    try:
        Stroke(0, 0, .1, 0, 0, 4.1)
    except ValueError:
        pass
    else:
        raise AssertionError("force outside action bounds must fail")
