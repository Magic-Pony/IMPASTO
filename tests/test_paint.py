import numpy as np

from impasto.sim.paint import BrushState, Canvas


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
