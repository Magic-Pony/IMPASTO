import numpy as np
import pytest

from scripts.run_stage3 import validate_palette_contract
from impasto.stage3.planner import AdaptiveMPCPlanner, _project_action
from impasto.stage3.splitter import split_difference


def test_pca_ransac_splitter_recovers_a_curve_and_color():
    height = width = 128
    base = np.ones((height, width), dtype=np.float32)
    target = base.copy()
    target[62:67, 20:108] = 0.2
    base_rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=-1)
    target_rgb = base_rgb.copy()
    target_rgb[62:67, 20:108] = np.array([217, 61, 54], dtype=np.uint8)
    proposals = split_difference(
        base, target, components=1, threshold=.05, min_pixels=20,
        kmeans_iterations=5, ransac_iterations=20, ransac_threshold_px=2.5,
        canvas_width=.4572, canvas_height=.5334,
        color_subset=[0, 1, 7, 12], base_rgb=base_rgb, target_rgb=target_rgb,
        seed=8)
    assert proposals
    proposal = proposals[0]
    assert proposal.model in {"line", "quadratic"}
    assert proposal.action[2] > .025
    assert proposal.color == 1
    assert proposal.mask.any()


def test_project_action_keeps_full_bezier_inside_workspace_and_canvas():
    bounds = {
        "x_m": [-.11, .04], "y_m": [-.11, .11], "length_m": [.025, .085],
        "bend_m": [-.035, .035], "angle_rad": [-np.pi, np.pi],
        "force_N": [.4, 1.5],
    }
    action = np.array([.039, .109, .08, .03, np.pi / 2, 1.0], dtype=np.float32)
    projected = _project_action(action, bounds, (.15, .16), (.4572, .5334))
    assert projected is not None
    from impasto.actions import Stroke
    points = Stroke(*map(float, projected)).points(np.linspace(0, 1, 200))
    assert np.max(np.abs(points[:, 0])) <= .15 + 1e-7
    assert np.max(np.abs(points[:, 1])) <= .16 + 1e-7
    assert np.max(np.abs(points[:, 0])) <= .4572 / 2 + 1e-7
    assert np.max(np.abs(points[:, 1])) <= .5334 / 2 + 1e-7


def test_overpaint_objective_penalizes_extra_pigment_without_changing_default():
    planner = AdaptiveMPCPlanner(_ToyDynamics(), {
        "seed": 1, "action_bounds": {}, "planner": {},
    })
    goal = np.array([[.5, .5]], dtype=np.float32)
    weights = np.ones_like(goal)
    assert planner.cost(np.array([[.3, .7]]), goal, weights) == pytest.approx(.2)
    planner.pc["overpaint_weight"] = 3.0
    assert planner.cost(np.array([[.3, .7]]), goal, weights) == pytest.approx(.4)


class _ToyDynamics:
    canvas_width = .4572
    canvas_height = .5334
    workspace_limits = (.15, .16)

    def rollout_batch(self, initial, actions, colors, opacities=None):
        # A deterministic image-space surrogate: each action paints one pixel.
        states = np.asarray(initial, dtype=np.float32).copy()
        frames = []
        for t in range(actions.shape[1]):
            x = np.clip(((actions[:, t, 0] + .11) / .15 * 31).astype(int), 0, 31)
            y = np.clip(((.11 - actions[:, t, 1]) / .22 * 31).astype(int), 0, 31)
            states[np.arange(len(states)), y, x] = .1
            frames.append(states.copy())
        return states, frames

    def rollout(self, initial, actions, colors, opacities=None):
        states, frames = self.rollout_batch(np.asarray(initial)[None],
                                            np.asarray(actions)[None],
                                            np.asarray(colors)[None], None)
        return states[0], [frame[0] for frame in frames]


def test_planner_returns_actions_matching_best_prediction():
    config = {
        "seed": 7,
        "action_bounds": {
            "x_m": [-.11, .04], "y_m": [-.11, .11], "length_m": [.025, .085],
            "bend_m": [-.035, .035], "angle_rad": [-np.pi, np.pi],
            "force_N": [.4, 1.5],
        },
        "planner": {
            "horizon": 2, "candidates": 8, "iterations": 2, "elite_ratio": .25,
            "temperature": .02, "ema": .8,
            "initial_sigma": [.012, .012, .012, .009, .22, .45],
            "minimum_sigma": [.002, .002, .004, .002, .04, .08],
            "maximum_sigma": [.03, .03, .035, .02, .45, 1.0],
            "covariance_floor": 1e-7, "objective_threshold": .015,
            "objective_dilation_kernel": 3, "objective_changed_weight": 4.0,
            "splitter_components": 2, "splitter_kmeans_iterations": 5,
            "splitter_ransac_iterations": 10, "splitter_ransac_threshold_px": 2.5,
            "splitter_min_pixels": 4,
        },
        "color_subset": [0, 1, 7, 12],
    }
    dynamics = _ToyDynamics()
    planner = AdaptiveMPCPlanner(dynamics, config)
    current = np.ones((32, 32), dtype=np.float32)
    goal = current.copy()
    goal[15:17, 15:17] = .1
    result = planner.plan(current, goal)
    replay, _ = dynamics.rollout(current, result.actions, result.colors)
    np.testing.assert_allclose(result.predicted, replay)
    assert result.final_cost <= result.initial_cost + 1e-7


def test_palette_contract_rejects_checkpoint_with_swapped_slots():
    palette = [[i, i, i] for i in range(24)]
    checkpoint_palette = [row.copy() for row in palette]
    checkpoint_palette[22], checkpoint_palette[23] = checkpoint_palette[23], checkpoint_palette[22]
    with pytest.raises(ValueError, match="checkpoint"):
        validate_palette_contract(
            {"paint": {"palette_rgb": palette}},
            {"palette_rgb": palette},
            {"deployment_palette_rgb": palette,
             "color_database": {"palette_rgb": palette}},
            {"config": {"palette_rgb": checkpoint_palette}},
        )


def test_robust_splitter_keeps_connected_different_colors_separate():
    import cv2
    from impasto.actions import Stroke

    height = width = 128
    base_rgb = np.full((height, width, 3), 245, dtype=np.uint8)
    target_rgb = base_rgb.copy()
    cv2.line(target_rgb, (20, 36), (62, 64), (64, 113, 175), 11)
    cv2.line(target_rgb, (62, 64), (62, 113), (80, 160, 80), 11)
    base = np.full((height, width), 1.0, dtype=np.float32)
    target = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    proposals = split_difference(
        base, target, components=3, threshold=.02, min_pixels=10,
        kmeans_iterations=8, ransac_iterations=16,
        canvas_width=.4572, canvas_height=.5334,
        color_subset=[0, 1], base_rgb=base_rgb, target_rgb=target_rgb,
        palette=[[64, 113, 175], [80, 160, 80]], robust_candidates=True, seed=7,
    )
    assert proposals
    for proposal in proposals:
        points = Stroke(*map(float, proposal.action)).points(np.linspace(0, 1, 70))
        x = np.clip(np.rint((points[:, 0] / .4572 + .5) * 127).astype(int), 0, 127)
        y = np.clip(np.rint((.5 - points[:, 1] / .5334) * 127).astype(int), 0, 127)
        on_blue = np.linalg.norm(target_rgb[y, x].astype(float) - [64, 113, 175], axis=1) < 10
        on_green = np.linalg.norm(target_rgb[y, x].astype(float) - [80, 160, 80], axis=1) < 10
        assert min(on_blue.sum(), on_green.sum()) <= 5
