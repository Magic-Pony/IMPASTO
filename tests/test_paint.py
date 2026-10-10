import numpy as np

from impasto.sim.paint import BrushState, Canvas, footprint_weight


def test_contact_and_pigment_are_required_for_deposition():
    c = Canvas(resolution=128, seed=4)
    b = BrushState(); b.dip(0)
    c.update([0.0, 0.0], 0.8, b, .01, contacting=False)
    assert not c.changed_mask().any()
    c.update([0.0, 0.0], 0.8, b, .01, contacting=True)
    c.update([0.03, 0.0], 0.8, b, .01, contacting=True)
    assert c.changed_mask().any()


def test_same_seed_gives_same_canvas_tooth():
    a, b = Canvas(resolution=64, seed=11), Canvas(resolution=64, seed=11)
    np.testing.assert_array_equal(a.tooth, b.tooth)


def test_world_positive_y_maps_to_image_top():
    c = Canvas(resolution=128)
    assert c.pixel([0.0, 0.10])[1] < c.pixel([0.0, -0.10])[1]


def test_gaussian_footprint_has_smooth_monotone_edge_and_finite_support():
    d2 = np.linspace(0.0, 1.1, 1001)
    weight = footprint_weight(d2, {
        "footprint_profile": "gaussian",
        "footprint_sigma_fraction": .72,
        "footprint_taper_start_fraction": .75,
    })
    assert weight[0] == 1.0
    assert np.all(np.diff(weight) <= 1e-7)
    assert weight[np.searchsorted(d2, 1.0)] == 0.0
    assert np.all(weight[d2 >= 1.0] == 0.0)


def test_legacy_linear_footprint_remains_the_default():
    d2 = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(footprint_weight(d2, {}), [1.0, 1.0, 1.0, 0.0])
