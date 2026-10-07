import numpy as np
import torch

from impasto.actions import Stroke
from impasto.stage2.data import transform_patch_action, weighted_l1_weights
from impasto.stage2.model import ImpastoUNet
from impasto.stage2.raster import (CropTransform, crop_canvas_gray, crop_for_stroke,
                                   normalize_patch_action, patch_action,
                                   render_stroke_patch)


def curve_from_patch_action(action, s):
    p0 = action[:2]
    alpha = action[5]
    tangent = np.array([np.cos(alpha), np.sin(alpha)])
    normal = np.array([-tangent[1], tangent[0]])
    return p0 + s * action[2] * tangent + 2 * s * (1 - s) * action[3] * normal


def test_world_stroke_maps_to_same_local_raster_curve():
    stroke = Stroke(-.03, .02, .065, -.021, .72, .83, color=7)
    crop = crop_for_stroke(stroke, patch_size=100, margin_m=.025)
    local = patch_action(stroke, crop)
    s = np.linspace(0, 1, 20)
    expected = crop.world_to_patch(stroke.points(s))
    actual = curve_from_patch_action(local, s[:, None])
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    encoded = normalize_patch_action(local, [.4, 1.5], 100)
    assert encoded.shape == (6,) and np.all((encoded >= 0) & (encoded <= 1))
    opposite = Stroke(-.03, .02, .065, .021, .72, .83, color=7)
    opposite_local = patch_action(opposite, crop)
    opposite_encoded = normalize_patch_action(opposite_local, [.4, 1.5], 100)
    assert (encoded[3] - .5) * (opposite_encoded[3] - .5) < 0


def test_bezier_action_augmentation_matches_transformed_curve():
    stroke = Stroke(-.03, .02, .065, -.021, .72, .83, color=7)
    crop = crop_for_stroke(stroke, patch_size=100, margin_m=.025)
    action = patch_action(stroke, crop)
    samples = np.linspace(0, 1, 20)
    points = np.stack([curve_from_patch_action(action, s) for s in samples])
    angle, shift = 3.2, np.array([1.5, -2.0])
    transformed_action = transform_patch_action(action, 100, angle, True, False, shift)
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    center = np.array([49.5, 49.5])
    expected = points - center
    expected = expected @ rotation.T
    expected[:, 0] *= -1
    expected += center + shift
    actual = np.stack([curve_from_patch_action(transformed_action, s) for s in samples])
    np.testing.assert_allclose(actual, expected, atol=2e-4)


def test_canvas_crop_and_action_raster_keep_luminance_range():
    rgb = np.full((512, 512, 3), [245, 242, 234], dtype=np.uint8)
    crop = CropTransform(0.0, 0.0, .12, 100)
    gray = crop_canvas_gray(rgb, crop)
    stroke = Stroke(-.02, -.01, .06, .01, .3, 1.0, color=1)
    rendered = render_stroke_patch(stroke, crop, {"minimum": .0018, "scale": .0042, "exponent": .55})
    assert gray.shape == rendered.shape == (100, 100)
    assert .9 < gray.mean() < 1.0
    assert rendered.min() < .5 and rendered.max() == 1.0


def test_weighted_l1_emphasizes_changed_region():
    current = torch.zeros(2, 1, 16, 16)
    target = current.clone()
    target[:, :, 7:9, 7:9] = .5
    weights = weighted_l1_weights(current, target, threshold=.1, dilation_kernel=5, changed_weight=4)
    assert torch.allclose(weights.sum(dim=(1, 2, 3)), torch.ones(2))
    assert weights[0, 0, 7, 7] > weights[0, 0, 0, 0]


def test_impasto_unet_predicts_full_next_state_patch():
    torch.set_num_threads(1)
    model = ImpastoUNet(dropout=0.0)
    image = torch.rand(2, 2, 100, 100)
    action = torch.rand(2, 6)
    prediction = model(image, action)
    assert prediction.shape == (2, 1, 100, 100)
    assert torch.isfinite(prediction).all()
    assert prediction.min() >= 0 and prediction.max() <= 1
    prediction.mean().backward()
    assert model.action_encoder[0].weight.grad is not None
