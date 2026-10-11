import numpy as np
import cv2

from impasto.stage3.planner import AdaptiveMPCPlanner
from impasto.stage3.splitter import StrokeProposal, split_difference


class _TinyWorkspace:
    canvas_width = 0.4572
    canvas_height = 0.5334
    workspace_limits = (0.03, 0.03)


def test_seed_masks_follow_executable_seeds_after_filter_and_padding(monkeypatch):
    bounds = {
        "x_m": [-0.01, 0.01], "y_m": [-0.01, 0.01],
        "length_m": [0.025, 0.085], "bend_m": [-0.035, 0.035],
        "angle_rad": [-np.pi, np.pi], "force_N": [0.4, 1.5],
    }
    invalid_mask = np.zeros((20, 20), dtype=bool)
    invalid_mask[1:4, 1:4] = True
    valid_mask = np.zeros((20, 20), dtype=bool)
    valid_mask[12:16, 12:16] = True
    proposals = [
        StrokeProposal(np.array([-0.01, 0.0, 0.085, 0.0, 0.0, 0.8]),
                       2, 1.0, invalid_mask, 1.0, "line"),
        StrokeProposal(np.array([-0.01, 0.0, 0.025, 0.0, 0.0, 0.8]),
                       7, 1.0, valid_mask, 1.0, "line"),
    ]
    monkeypatch.setattr("impasto.stage3.planner.split_difference",
                        lambda *args, **kwargs: proposals)
    config = {
        "seed": 1,
        "action_bounds": bounds,
        "planner": {
            "splitter_components": 2, "objective_threshold": 0.02,
            "splitter_min_pixels": 4, "splitter_kmeans_iterations": 2,
            "splitter_ransac_iterations": 2,
            "splitter_ransac_threshold_px": 2.5,
        },
    }
    planner = AdaptiveMPCPlanner(_TinyWorkspace(), config)
    current = np.ones((20, 20), dtype=np.float32)
    actions, colors, opacities, returned, masks = planner._initial_sequence(
        current, current, horizon=2)

    assert returned is proposals
    assert actions.shape == (2, 6)
    assert colors == [7, 7]
    assert opacities == [1.0, 1.0]
    np.testing.assert_array_equal(masks[0], valid_mask)
    np.testing.assert_array_equal(masks[1], valid_mask)


def test_generic_thin_stroke_pool_preserves_width_alternatives():
    base_rgb = np.full((96, 96, 3), 245, dtype=np.uint8)
    target_rgb = base_rgb.copy()
    pigment = (130, 185, 140)
    cv2.line(target_rgb, (42, 20), (42, 75), pigment, 3)
    cv2.line(target_rgb, (54, 20), (54, 62), pigment, 3)
    gray = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    bounds = {
        "x_m": [-.125, .07], "y_m": [-.135, .135],
        "length_m": [.004, .105], "bend_m": [-.05, .05],
        "force_N": [.4, 1.5],
    }
    pool = split_difference(
        gray(base_rgb), gray(target_rgb), components=4, threshold=.01,
        min_pixels=5, kmeans_iterations=8, ransac_iterations=16,
        canvas_width=.4572, canvas_height=.5334, action_bounds=bounds,
        seed=7, color_subset=[0], base_rgb=base_rgb, target_rgb=target_rgb,
        palette=[pigment], opaque_colors=True, robust_candidates=True,
        preserve_thin_regions=True, return_pool=True,
        force_hypotheses=[.4, .7, 1.0, 1.3])
    assert len(pool) >= 8
    assert {round(float(proposal.action[5]), 1) for proposal in pool} >= {
        .4, .7, 1.0, 1.3}
    assert len({round(float(proposal.action[2]), 3) for proposal in pool}) >= 2
