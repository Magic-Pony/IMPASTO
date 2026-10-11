"""Stage-3 learned-dynamics MPC with PCA-RANSAC and CMA-style sampling."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch

from impasto.actions import Stroke
from impasto.stage2.model import ImpastoUNet
from impasto.stage2.raster import (CropTransform, crop_canvas_gray, crop_for_stroke,
                                   normalize_patch_action, render_stroke_patch)
from impasto.stage3.splitter import StrokeProposal, split_difference


@dataclass
class PlanResult:
    actions: np.ndarray
    colors: list[int]
    opacities: list[float]
    predicted: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: list[dict]
    proposals: list[StrokeProposal]
    initial_actions: np.ndarray | None = None
    initial_colors: list[int] | None = None
    initial_masks: np.ndarray | None = None


def _crop_box(crop: CropTransform, width, height, canvas_width, canvas_height):
    cx = (crop.center_x_m / canvas_width + .5) * (width - 1)
    cy = (.5 - crop.center_y_m / canvas_height) * (height - 1)
    half_w = crop.side_m / canvas_width * (width - 1) / 2
    half_h = crop.side_m / canvas_height * (height - 1) / 2
    x0, x1 = int(round(cx - half_w)), int(round(cx + half_w))
    y0, y1 = int(round(cy - half_h)), int(round(cy + half_h))
    return max(0, x0), min(width, x1), max(0, y0), min(height, y1)


def _clip_action(action, bounds):
    action = np.asarray(action, dtype=np.float32).copy()
    for index, name in enumerate(("x_m", "y_m", "length_m", "bend_m")):
        action[index] = np.clip(action[index], *bounds[name])
    action[4] = ((action[4] + np.pi) % (2 * np.pi)) - np.pi
    action[5] = np.clip(action[5], *bounds["force_N"])
    return action


def _wrap_angle(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _action_delta(action, reference):
    delta = np.asarray(action, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    delta[4] = _wrap_angle(delta[4])
    return delta


def _clip_to_seed(action, seed, trust_region, bounds):
    delta = _action_delta(action, seed)
    delta = np.clip(delta, -np.asarray(trust_region), np.asarray(trust_region))
    constrained = np.asarray(seed, dtype=np.float32) + delta.astype(np.float32)
    constrained[4] = _wrap_angle(seed[4] + delta[4])
    return _clip_action(constrained, bounds)


def _weighted_action_mean(actions, weights, reference):
    result = np.sum(np.asarray(actions, dtype=np.float64)
                    * np.asarray(weights, dtype=np.float64)[:, None], axis=0)
    angles = np.asarray(actions, dtype=np.float64)[:, 4]
    sin_mean = float(np.sum(np.asarray(weights) * np.sin(angles)))
    cos_mean = float(np.sum(np.asarray(weights) * np.cos(angles)))
    if abs(sin_mean) + abs(cos_mean) > 1e-9:
        result[4] = np.arctan2(sin_mean, cos_mean)
    else:
        result[4] = float(reference[4])
    result[4] = _wrap_angle(result[4])
    return result


def _project_action(action, bounds, workspace_limits, canvas_size):
    action = _clip_action(action, bounds)
    probe = Stroke(*map(float, action))
    relative = probe.points(np.linspace(0.0, 1.0, 129)) - np.array([probe.x, probe.y])
    lows, highs = [], []
    for axis, name, workspace, half_canvas in (
            (0, "x_m", workspace_limits[0], canvas_size[0] / 2),
            (1, "y_m", workspace_limits[1], canvas_size[1] / 2)):
        low = max(float(bounds[name][0]), -workspace - float(relative[:, axis].min()),
                  -half_canvas - float(relative[:, axis].min()))
        high = min(float(bounds[name][1]), workspace - float(relative[:, axis].max()),
                   half_canvas - float(relative[:, axis].max()))
        if low > high:
            return None
        lows.append(low)
        highs.append(high)
        action[axis] = np.clip(action[axis], low, high)
    projected = Stroke(*map(float, action)).points(np.linspace(0.0, 1.0, 129))
    if (np.any(np.abs(projected[:, 0]) > workspace_limits[0] + 1e-7)
            or np.any(np.abs(projected[:, 1]) > workspace_limits[1] + 1e-7)
            or np.any(np.abs(projected[:, 0]) > canvas_size[0] / 2 + 1e-7)
            or np.any(np.abs(projected[:, 1]) > canvas_size[1] / 2 + 1e-7)):
        return None
    return action


class LearnedCanvasDynamics:
    """Adapt Stage-2 local patch prediction to a global canvas rollout."""

    def __init__(self, model, device, stage2_config, stage3_config):
        self.cfg = stage3_config
        self.model = model.eval()
        self.device = device
        self.stage2 = stage2_config
        self.canvas_width = float(stage3_config["canvas_width_m"])
        self.canvas_height = float(stage3_config["canvas_height_m"])
        self.patch_size = int(stage3_config["patch_size"])
        self.radius_cfg = stage3_config["force_radius_m"]
        self.margin = float(stage2_config["crop_margin_m"])
        self.workspace_limits = tuple(map(float, stage2_config.get("workspace_abs_limit_m", [.15, .16])))

    def effective_opacity(self, opacity):
        """Map Appendix-B's image ratio to the local simulator's paint load.

        The ratio is measured from a low-resolution camera image, whereas
        Canvas applies it to a fully loaded brush footprint.  A short
        calibration factor keeps the learned grayscale rollout and MuJoCo
        execution on the same observed intensity scale.
        """
        scale = float(self.cfg.get("opacity_scale", 1.0))
        return float(np.clip(float(opacity) * scale, 0.0, 1.0))

    def predict_local(self, canvas_gray, action, color, opacity=1.0):
        # Stage 2 was trained on opaque self-play strokes. Scale its predicted
        # state change by the paper's transparency ratio after inference.
        stroke = Stroke(*map(float, action), int(color))
        crop = crop_for_stroke(stroke, self.patch_size, self.margin)
        current = crop_canvas_gray(canvas_gray, crop, self.canvas_width, self.canvas_height)
        rendered = render_stroke_patch(stroke, crop, self.radius_cfg,
                                       palette=self.cfg.get("palette_rgb"))
        local_action = normalize_patch_action(
            self._local_action(stroke, crop), self.stage2["action_bounds"]["force_N"], self.patch_size)
        image = torch.from_numpy(np.stack([current, rendered])[None].astype(np.float32)).to(self.device)
        action_tensor = torch.from_numpy(local_action[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            prediction = self.model(image, action_tensor)[0, 0].detach().cpu().numpy()
        return prediction, crop, rendered

    def prepare_local(self, canvas_gray, action, color, opacity=1.0):
        stroke = Stroke(*map(float, action), int(color))
        crop = crop_for_stroke(stroke, self.patch_size, self.margin)
        current = crop_canvas_gray(canvas_gray, crop, self.canvas_width, self.canvas_height)
        rendered = render_stroke_patch(stroke, crop, self.radius_cfg,
                                       palette=self.cfg.get("palette_rgb"))
        local_action = normalize_patch_action(
            self._local_action(stroke, crop), self.stage2["action_bounds"]["force_N"], self.patch_size)
        return np.stack([current, rendered]), local_action, crop

    @staticmethod
    def _local_action(stroke, crop):
        p0 = crop.world_to_patch([stroke.x, stroke.y])
        alpha_image = -float(stroke.angle)
        bend_image = -float(stroke.bend) / crop.side_m * (crop.patch_size - 1)
        length_pixels = float(stroke.length) / crop.side_m * (crop.patch_size - 1)
        return np.array([p0[0], p0[1], length_pixels, bend_image,
                         float(stroke.force), alpha_image], dtype=np.float32)

    def apply_local(self, canvas_gray, prediction, crop, opacity=1.0):
        result = np.asarray(canvas_gray, dtype=np.float32).copy()
        x0, x1, y0, y1 = _crop_box(crop, result.shape[1], result.shape[0],
                                   self.canvas_width, self.canvas_height)
        if x1 <= x0 or y1 <= y0:
            return result
        resized = cv2.resize(prediction, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        current = np.asarray(canvas_gray, dtype=np.float32)[y0:y1, x0:x1]
        alpha = self.effective_opacity(opacity)
        if self.cfg.get("residual_patch_paste", False):
            local_current = crop_canvas_gray(canvas_gray, crop, self.canvas_width, self.canvas_height)
            delta = cv2.resize(prediction-local_current, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
            mixed = current + alpha * delta
        else:
            mixed = current + alpha * (resized - current)
        result[y0:y1, x0:x1] = np.clip(mixed, 0.0, 1.0)
        return result

    def step(self, canvas_gray, action, color, opacity=1.0):
        prediction, crop, rendered = self.predict_local(canvas_gray, action, color, opacity)
        return self.apply_local(canvas_gray, prediction, crop, opacity), prediction, crop, rendered

    def rollout(self, initial, actions, colors, opacities=None):
        canvases, frames = self.rollout_batch(np.asarray(initial, dtype=np.float32)[None],
                                               np.asarray(actions, dtype=np.float32)[None],
                                               np.asarray(colors, dtype=np.int64)[None],
                                               None if opacities is None else np.asarray(opacities, dtype=np.float32)[None])
        return canvases[0], [frame[0] for frame in frames]

    def rollout_batch(self, initial, actions, colors, opacities=None):
        states = np.asarray(initial, dtype=np.float32).copy()
        actions = np.asarray(actions, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.int64)
        opacities = (np.ones_like(colors, dtype=np.float32) if opacities is None
                     else np.asarray(opacities, dtype=np.float32))
        batch_size, horizon = actions.shape[:2]
        frames = []
        for t in range(horizon):
            local_inputs, local_actions, crops = [], [], []
            for i in range(batch_size):
                image, action, crop = self.prepare_local(
                    states[i], actions[i, t], colors[i, t], opacities[i, t])
                local_inputs.append(image)
                local_actions.append(action)
                crops.append(crop)
            image_tensor = torch.from_numpy(np.asarray(local_inputs, dtype=np.float32)).to(self.device)
            action_tensor = torch.from_numpy(np.asarray(local_actions, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                predictions = self.model(image_tensor, action_tensor)[:, 0].cpu().numpy()
            for i in range(batch_size):
                states[i] = self.apply_local(states[i], predictions[i], crops[i], opacities[i, t])
            frames.append(states.copy())
        return states, frames


class AdaptiveMPCPlanner:
    """Receding-horizon planner matching Algorithm 1's control flow."""

    def __init__(self, dynamics, stage3_config):
        self.dynamics = dynamics
        self.cfg = stage3_config
        self.bounds = stage3_config["action_bounds"]
        self.pc = stage3_config["planner"]
        self.rng = np.random.default_rng(int(stage3_config["seed"]))
        self.color_target = None
        self.color_current = None

    def objective_weights(self, current, goal):
        diff = np.abs(np.asarray(goal) - np.asarray(current))
        mask = (diff > float(self.pc["objective_threshold"])).astype(np.uint8)
        kernel = int(self.pc["objective_dilation_kernel"])
        dilated = cv2.dilate(mask, np.ones((kernel, kernel), np.uint8))
        return 1.0 + float(self.pc["objective_changed_weight"]) * dilated.astype(np.float32)

    def cost(self, prediction, goal, weights):
        return float(np.sum(weights * np.abs(prediction - goal)) / max(float(weights.sum()), 1e-8))

    def _random_action(self):
        b = self.bounds
        for _ in range(256):
            action = np.array([
                self.rng.uniform(*b["x_m"]), self.rng.uniform(*b["y_m"]),
                self.rng.uniform(*b["length_m"]), self.rng.uniform(*b["bend_m"]),
                self.rng.uniform(*b["angle_rad"]), self.rng.uniform(*b["force_N"]),
            ], dtype=np.float32)
            valid = _project_action(action, b, self.dynamics.workspace_limits,
                                    (self.dynamics.canvas_width, self.dynamics.canvas_height))
            if valid is not None:
                return valid
        fallback = np.array([-.06, -.06, .03, 0.0, 0.0, .8], dtype=np.float32)
        return _project_action(fallback, b, self.dynamics.workspace_limits,
                               (self.dynamics.canvas_width, self.dynamics.canvas_height))

    def _initial_sequence(self, current, goal, horizon, color_images=None,
                          proposal_images=None):
        proposal_base = current if proposal_images is None else proposal_images[0]
        # Geometry may use a published prefix frame, while colour and alpha
        # must be estimated from the actually observed canvas.  The prefix is
        # only a target-frame alignment aid; feeding it as the RGB base makes
        # all antialiasing and registration differences look like pigment.
        proposal_base_rgb = (color_images[0] if color_images is not None else None)
        proposals = split_difference(
            proposal_base, goal,
            components=min(int(self.pc["splitter_components"]), int(horizon)),
            threshold=float(self.pc.get("splitter_threshold", self.pc["objective_threshold"])),
            min_pixels=int(self.pc["splitter_min_pixels"]),
            kmeans_iterations=int(self.pc["splitter_kmeans_iterations"]),
            ransac_iterations=int(self.pc["splitter_ransac_iterations"]),
            ransac_threshold_px=float(self.pc["splitter_ransac_threshold_px"]),
            color_guided=bool(self.pc.get("splitter_color_guided", False)),
            opaque_colors=bool(self.cfg.get("opaque_colors", False)),
            robust_candidates=bool(self.pc.get("robust_candidates", False)),
            radius_config=self.cfg.get("force_radius_m"),
            curvature_penalty=float(self.pc.get("splitter_curvature_penalty", .01)),
            overlap_penalty=float(self.pc.get("splitter_overlap_penalty", .15)),
            preserve_thin_regions=bool(self.pc.get("splitter_preserve_thin_regions", False)),
            foreground_threshold=float(self.pc.get("splitter_foreground_threshold", .06)),
            return_pool=bool(self.pc.get("learned_seed_selection", False)),
            force_hypotheses=self.pc.get("seed_force_hypotheses"),
            canvas_width=self.dynamics.canvas_width,
            canvas_height=self.dynamics.canvas_height,
            action_bounds=self.bounds,
            seed=int(self.cfg["seed"]),
            color_subset=self.cfg.get("active_color_subset", self.cfg.get("color_subset")),
            base_rgb=proposal_base_rgb,
            target_rgb=None if color_images is None else color_images[1],
            palette=self.cfg.get("palette_rgb"),
        )
        if self.pc.get("learned_seed_selection", False) and proposals:
            proposals = self._select_learned_seeds(current, goal, proposals, horizon,
                                                   color_images)
        actions = []
        colors = []
        opacities = []
        masks = []
        for proposal in proposals[:horizon]:
            action = _project_action(proposal.action, self.bounds, self.dynamics.workspace_limits,
                                     (self.dynamics.canvas_width, self.dynamics.canvas_height))
            if action is None:
                continue
            actions.append(action)
            colors.append(int(proposal.color))
            opacities.append(float(proposal.opacity))
            masks.append(np.asarray(proposal.mask, dtype=bool))
        while len(actions) < horizon:
            if not actions:
                if not bool(self.pc.get("allow_random_fill", True)):
                    raise ValueError("No supported stroke region for this planning window")
                actions.append(self._random_action())
                colors.append(colors[-1] if colors else 0)
                opacities.append(opacities[-1] if opacities else 1.0)
                masks.append(np.zeros_like(goal, dtype=bool))
                continue
            # A duplicate supported seed is safe for refinement. A random
            # workspace stroke is not: it can survive MPPI at the edge of a
            # sparse objective and permanently damage the canvas.
            source = (len(actions) - 1) % len(actions)
            actions.append(actions[source].copy())
            colors.append(colors[source])
            opacities.append(opacities[source])
            masks.append(masks[source].copy())
        return (np.asarray(actions, dtype=np.float32), colors, opacities, proposals,
                np.asarray(masks, dtype=bool))

    def _rgb_step(self, rgb, gray, next_gray, colors):
        palette = np.asarray(self.cfg["palette_rgb"], np.float32) / 255.0
        pigment = palette[np.asarray(colors, dtype=int)]
        pigment_gray = pigment @ np.array([.299, .587, .114], np.float32)
        denominator = pigment_gray[:, None, None] - gray
        alpha = np.divide(next_gray - gray, denominator,
                          out=np.zeros_like(gray), where=abs(denominator) > .025)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha[abs(next_gray - gray) < .003] = 0.0
        return rgb + alpha[..., None] * (pigment[:, None, None, :] - rgb)

    def _select_learned_seeds(self, current, goal, proposals, horizon, color_images):
        """Select initial stroke hypotheses with the deployed dynamics model.

        The splitter supplies alternative decompositions and force values. A
        greedy rollout keeps a small residual stroke from losing to a wide
        stroke merely because the latter explains more pixels in the seed
        mask. This ranking is target agnostic and uses only the current goal.
        """
        projected = []
        for proposal in proposals:
            action = _project_action(proposal.action, self.bounds,
                                     self.dynamics.workspace_limits,
                                     (self.dynamics.canvas_width,
                                      self.dynamics.canvas_height))
            if action is not None:
                projected.append((proposal, action))
        if not projected:
            return []
        actions = np.asarray([action for _, action in projected], dtype=np.float32)[:, None]
        default_colors = np.asarray([[proposal.color] for proposal, _ in projected], dtype=np.int64)
        if self.color_target is not None:
            colors = self._candidate_colors(actions, [0])
        else:
            colors = default_colors
        weights = self.objective_weights(current, goal)
        state = np.asarray(current, dtype=np.float32).copy()
        rgb = (None if color_images is None else
               np.asarray(color_images[0], np.float32) / 255.0)
        target_rgb = self.color_target
        remaining = list(range(len(projected)))
        selected = []
        for _ in range(min(int(horizon), len(remaining))):
            batch_actions = actions[remaining]
            batch_colors = colors[remaining]
            initial = np.repeat(state[None], len(remaining), axis=0)
            predictions, frames = self.dynamics.rollout_batch(
                initial, batch_actions, batch_colors,
                np.ones((len(remaining), 1), dtype=np.float32))
            costs = np.asarray([self.cost(prediction, goal, weights)
                                for prediction in predictions], dtype=np.float32)
            predicted_rgb = None
            if (rgb is not None and target_rgb is not None
                    and float(self.pc.get("rgb_objective_weight", 0))):
                predicted_rgb = self._rgb_step(
                    np.repeat(rgb[None], len(remaining), axis=0),
                    np.repeat(state[None], len(remaining), axis=0),
                    frames[0], batch_colors[:, 0])
                rgb_error = np.mean(abs(predicted_rgb - target_rgb), axis=-1)
                costs += float(self.pc["rgb_objective_weight"]) * np.asarray([
                    np.sum(weights * error) / max(float(weights.sum()), 1e-8)
                    for error in rgb_error])
            best_local = int(np.argmin(costs))
            best_index = remaining.pop(best_local)
            proposal, action = projected[best_index]
            selected.append(StrokeProposal(
                action, int(batch_colors[best_local, 0]), 1.0, proposal.mask,
                float(-costs[best_local]), proposal.model))
            state = predictions[best_local]
            if predicted_rgb is not None:
                rgb = predicted_rgb[best_local]
        return selected

    def _sample(self, nominal, seed_actions, sigma, covariance_by_step, count):
        horizon = nominal.shape[0]
        candidates = np.empty((count, horizon, 6), dtype=np.float32)
        candidates[0] = nominal
        trust_region = self.pc.get("seed_trust_region")
        for i in range(1, count):
            for t in range(horizon):
                covariance = np.diag(sigma[t]) @ covariance_by_step[t] @ np.diag(sigma[t])
                covariance = covariance + np.eye(6) * float(self.pc["covariance_floor"])
                try:
                    chol = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError:
                    chol = np.diag(sigma[t])
                sampled = nominal[t] + chol @ self.rng.standard_normal(6)
                if trust_region is not None:
                    sampled = _clip_to_seed(sampled, seed_actions[t], trust_region,
                                            self.bounds)
                else:
                    sampled = _clip_action(sampled, self.bounds)
                projected = _project_action(
                    sampled, self.bounds, self.dynamics.workspace_limits,
                    (self.dynamics.canvas_width, self.dynamics.canvas_height))
                candidates[i, t] = nominal[t] if projected is None else projected
        return candidates

    def _candidate_colors(self, candidates, default):
        if self.color_target is None:
            return np.tile(np.asarray(default, dtype=np.int64), (len(candidates), 1))
        rgb = self.color_target
        h, w = rgb.shape[:2]
        palette = np.asarray(self.cfg["palette_rgb"], dtype=np.float32)/255
        ids = self.cfg.get("active_color_subset", self.cfg["color_subset"])
        result = np.empty(candidates.shape[:2], dtype=np.int64)
        background=np.median(rgb.reshape(-1,3)[::50],axis=0)
        # Appendix B uses predicted footprint intersected with a central disk.
        # This sparse central-band quadrature approximates that image region.
        for i, sequence in enumerate(candidates):
            for j, action in enumerate(sequence):
                stroke = Stroke(*map(float, action))
                pts = stroke.points(np.linspace(.25,.75,11))
                normal = np.array([-np.sin(stroke.angle),np.cos(stroke.angle)])
                rc=self.cfg["force_radius_m"]
                radius=rc["minimum"]+rc["scale"]*stroke.force**rc["exponent"]
                pts=(pts[:,None,:]+normal[None,None,:]*np.array([-.4,0,.4])[None,:,None]*radius).reshape(-1,2)
                x=np.clip(np.rint((pts[:,0]/self.dynamics.canvas_width+.5)*(w-1)).astype(int),0,w-1)
                y=np.clip(np.rint((.5-pts[:,1]/self.dynamics.canvas_height)*(h-1)).astype(int),0,h-1)
                samples=rgb[y,x]
                # Robust representative in linear RGB. Reject paper background.
                keep=np.max(abs(samples-background),axis=-1)>.05
                if keep.any(): samples=samples[keep]
                linear=lambda x: np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
                mu=linear(samples).mean(axis=0)
                database=self.cfg.get("color_database")
                if database:
                    delta=mu-np.asarray(database["means_linear_rgb"])[ids]
                    inv=np.linalg.inv(np.asarray(database["covariances"])[ids])
                    cost=np.einsum("ni,nij,nj->n",delta,inv,delta)
                else:
                    delta=abs(linear(palette[ids,None])-linear(samples[None]))
                    cost=np.mean(delta,axis=(1,2))
                result[i,j]=ids[int(np.argmin(cost))]
        return result

    def _evaluate(self, current, goal, candidates, colors, opacities, weights):
        color_batch = self._candidate_colors(candidates, colors)
        opacity_batch = np.tile(np.asarray(opacities, dtype=np.float32), (len(candidates), 1))
        initial_batch = np.tile(np.asarray(current, dtype=np.float32)[None],
                                (len(candidates), 1, 1))
        final_predictions, rollout_frames = self.dynamics.rollout_batch(
            initial_batch, candidates, color_batch, opacity_batch)
        costs = np.asarray([self.cost(prediction, goal, weights)
                            for prediction in final_predictions], dtype=np.float32)
        rgb_weight = float(self.pc.get("rgb_objective_weight", 0))
        if rgb_weight and self.color_current is not None:
            # Recover a physically feasible RGB change from the learned gray
            # next-state difference and the separate, known pigment color.
            rgb = np.broadcast_to(self.color_current, (len(candidates),*self.color_current.shape)).copy()
            gray = np.broadcast_to(current, (len(candidates),*current.shape)).copy()
            palette = np.asarray(self.cfg["palette_rgb"],np.float32)/255
            for t, next_gray in enumerate(rollout_frames):
                pigment=palette[color_batch[:,t]]
                pg=pigment @ np.array([.299,.587,.114],np.float32)
                denominator=pg[:,None,None]-gray
                alpha=np.divide(next_gray-gray,denominator,
                    out=np.zeros_like(gray),where=abs(denominator)>.025)
                alpha=np.clip(alpha,0,1)
                alpha[abs(next_gray-gray)<.003]=0
                rgb+=alpha[...,None]*(pigment[:,None,None,:]-rgb)
                gray=next_gray
            rgb_error=np.mean(abs(rgb-self.color_target),axis=-1)
            costs += rgb_weight*np.sum(rgb_error*weights,axis=(1,2))/weights.sum()
        return costs, final_predictions

    def plan(self, current, goal, colors=None, initial=None, color_images=None,
             proposal_images=None, opacities=None):
        horizon = int(self.pc["horizon"])
        self.color_target = (np.asarray(color_images[1],np.float32)/255 if color_images is not None and self.pc.get("update_colors_with_geometry",False) else None)
        self.color_current = (np.asarray(color_images[0],np.float32)/255 if color_images is not None else None)
        if initial is None:
            nominal, inferred_colors, inferred_opacities, proposals, seed_masks = self._initial_sequence(
                current, goal, horizon, color_images, proposal_images)
        else:
            nominal = np.asarray(initial, dtype=np.float32).copy()
            inferred_colors = list(colors or [0] * horizon)
            inferred_opacities = list(opacities or [1.0] * horizon)
            proposals = []
            seed_masks = None
            if len(nominal) < horizon:
                pad = np.repeat(nominal[-1:, :], horizon - len(nominal), axis=0)
                nominal = np.concatenate([nominal, pad], axis=0)
            nominal = nominal[:horizon]
            inferred_colors = inferred_colors[:horizon]
        inferred_colors = self._candidate_colors(nominal[None], inferred_colors)[0].tolist()
        seed_actions = nominal.copy()
        seed_colors = list(inferred_colors)
        if colors is not None:
            inferred_colors = list(colors)
        if opacities is not None:
            inferred_opacities = list(opacities)
        while len(inferred_colors) < horizon:
            inferred_colors.append(inferred_colors[-1] if inferred_colors else 0)
        while len(inferred_opacities) < horizon:
            inferred_opacities.append(inferred_opacities[-1] if inferred_opacities else 1.0)
        inferred_colors = inferred_colors[:horizon]
        inferred_opacities = inferred_opacities[:horizon]
        sigma = np.tile(np.asarray(self.pc["initial_sigma"], dtype=np.float32), (horizon, 1))
        covariance = np.tile(np.eye(6, dtype=np.float64), (horizon, 1, 1))
        step_paths = np.zeros((horizon, 6), dtype=np.float64)
        covariance_paths = np.zeros((horizon, 6), dtype=np.float64)
        weights = self.objective_weights(current, goal)
        initial_prediction, _ = self.dynamics.rollout(
            current, nominal, inferred_colors, inferred_opacities)
        initial_cost = float(self._evaluate(current, goal, nominal[None], inferred_colors, inferred_opacities, weights)[0][0])
        trace = []
        best_actions = nominal.copy()
        best_colors = list(inferred_colors)
        best_prediction = initial_prediction
        best_cost = initial_cost
        for iteration in range(int(self.pc["iterations"])):
            candidates = self._sample(nominal, seed_actions, sigma, covariance,
                                      int(self.pc["candidates"]))
            costs, predictions = self._evaluate(
                current, goal, candidates, inferred_colors, inferred_opacities, weights)
            elite_count = max(2, int(float(self.pc["elite_ratio"]) * len(candidates)))
            elite_indices = np.argsort(costs)[:elite_count]
            elite = candidates[elite_indices]
            elite_costs = costs[elite_indices]
            temperature = max(float(self.pc["temperature"]), 1e-6)
            logits = -(elite_costs - float(elite_costs.min())) / temperature
            softmax = np.exp(logits - logits.max())
            softmax /= max(float(softmax.sum()), 1e-12)
            weighted_mean = np.stack([
                _weighted_action_mean(elite[:, t], softmax, nominal[t])
                for t in range(horizon)
            ]).astype(np.float32)
            old_nominal = nominal.copy()
            ema = float(self.pc["ema"])
            updated = []
            for t in range(horizon):
                action = old_nominal[t] + ema * _action_delta(weighted_mean[t], old_nominal[t])
                if self.pc.get("seed_trust_region") is not None:
                    action = _clip_to_seed(action, seed_actions[t],
                                           self.pc["seed_trust_region"], self.bounds)
                else:
                    action = _clip_action(action, self.bounds)
                projected = _project_action(
                    action, self.bounds, self.dynamics.workspace_limits,
                    (self.dynamics.canvas_width, self.dynamics.canvas_height))
                updated.append(seed_actions[t] if projected is None else projected)
            nominal = np.asarray(updated, dtype=np.float32)
            dimension = nominal.shape[-1]
            mu_eff = 1.0 / max(float(np.sum(softmax ** 2)), 1e-12)
            c_sigma = (mu_eff + 2.0) / (dimension + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dimension + 1.0)) - 1.0) + c_sigma
            c_c = (4.0 + mu_eff / dimension) / (dimension + 4.0 + 2.0 * mu_eff / dimension)
            c_1 = 2.0 / ((dimension + 1.3) ** 2 + mu_eff)
            c_mu = min(1.0 - c_1,
                       2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dimension + 2.0) ** 2 + mu_eff))
            chi_n = math.sqrt(dimension) * (1.0 - 1.0 / (4.0 * dimension)
                                            + 1.0 / (21.0 * dimension ** 2))
            for t in range(horizon):
                scaled_shift = _action_delta(weighted_mean[t], old_nominal[t]) / np.maximum(sigma[t], 1e-8)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance[t])
                inv_sqrt = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8)))) @ eigenvectors.T
                step_paths[t] = ((1.0 - c_sigma) * step_paths[t]
                                 + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
                                 * (inv_sqrt @ scaled_shift))
                path_norm = float(np.linalg.norm(step_paths[t]))
                correction = math.sqrt(max(1.0 - (1.0 - c_sigma) ** (2 * (iteration + 1)), 1e-8))
                h_sigma = float(path_norm / correction / chi_n
                                < 1.4 + 2.0 / (dimension + 1.0))
                covariance_paths[t] = ((1.0 - c_c) * covariance_paths[t]
                                       + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff)
                                       * scaled_shift)
                rank_mu = np.zeros((dimension, dimension), dtype=np.float64)
                for weight, member in zip(softmax, elite[:, t]):
                    delta = _action_delta(member, old_nominal[t]) / np.maximum(sigma[t], 1e-8)
                    rank_mu += float(weight) * np.outer(delta, delta)
                covariance[t] = ((1.0 - c_1 - c_mu) * covariance[t]
                                 + c_1 * np.outer(covariance_paths[t], covariance_paths[t])
                                 + c_mu * rank_mu)
                covariance[t] = .5 * (covariance[t] + covariance[t].T)
                eigvals, eigvecs = np.linalg.eigh(covariance[t])
                covariance[t] = (eigvecs * np.maximum(eigvals, float(self.pc["covariance_floor"]))) @ eigvecs.T
                step_factor = float(np.exp(np.clip((c_sigma / d_sigma)
                                                   * (path_norm / chi_n - 1.0), -0.5, 0.5)))
                sigma[t] = np.clip(sigma[t] * step_factor,
                                   np.asarray(self.pc["minimum_sigma"]),
                                   np.asarray(self.pc["maximum_sigma"]))
            best_index = int(np.argmin(costs))
            if float(costs[best_index]) < best_cost:
                best_cost = float(costs[best_index])
                best_actions = candidates[best_index].copy()
                best_colors = self._candidate_colors(best_actions[None],inferred_colors)[0].tolist()
                best_prediction = predictions[best_index]
            trace.append({"iteration": iteration, "best_cost": float(costs[best_index]),
                          "elite_cost": float(elite_costs.mean()),
                          "nominal_cost": float(self._evaluate(current,goal,nominal[None],inferred_colors,inferred_opacities,weights)[0][0]),
                          "sigma_mean": float(sigma.mean())})
        inferred_colors = self._candidate_colors(nominal[None],inferred_colors)[0].tolist()
        final_prediction, _ = self.dynamics.rollout(
            current, nominal, inferred_colors, inferred_opacities)
        final_cost = float(self._evaluate(current,goal,nominal[None],inferred_colors,inferred_opacities,weights)[0][0])
        if final_cost < best_cost:
            best_actions = nominal.copy()
            best_colors = list(inferred_colors)
            best_prediction, best_cost = final_prediction, final_cost
        return PlanResult(best_actions, best_colors, inferred_opacities,
                          best_prediction, initial_cost,
                          float(best_cost), trace, proposals, seed_actions, seed_colors,
                          seed_masks)
