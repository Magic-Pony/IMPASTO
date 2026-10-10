#!/usr/bin/env python3
"""Run single-stroke and receding-horizon Stage-3 painting experiments."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.sim.paint import Canvas
from impasto.stage2.model import ImpastoUNet
from impasto.stage2.raster import srgb_gray
from impasto.stage3.planner import AdaptiveMPCPlanner, LearnedCanvasDynamics
from impasto.stage3.targets import render_single_target, render_target_prefix


ROOT = Path(__file__).resolve().parents[1]


def validate_palette_contract(stage1, stage2, stage3, checkpoint):
    expected = np.asarray(stage1["paint"]["palette_rgb"])
    sources = {
        "training configuration": stage2.get("palette_rgb"),
        "checkpoint": checkpoint.get("config", {}).get("palette_rgb"),
        "deployment": stage3.get("deployment_palette_rgb", expected),
        "color database": stage3.get("color_database", {}).get("palette_rgb", expected),
    }
    for name, colors in sources.items():
        if colors is None or not np.array_equal(expected, np.asarray(colors)):
            raise ValueError(f"Palette mismatch in {name}; recollect/retrain or use its matching palette")


def load_system():
    config_name = os.environ.get("IMPASTO_STAGE3_CONFIG", "configs/stage3.json")
    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    stage3 = json.loads(config_path.read_text())
    stage3["config_file"] = str(config_path.relative_to(ROOT))
    if stage3.get("color_database_file"):
        stage3["color_database"] = json.loads((ROOT / stage3["color_database_file"]).read_text())
    stage2 = json.loads((ROOT / stage3["stage2_config"]).read_text())
    stage1 = json.loads((ROOT / stage3["stage1_config"]).read_text())
    if stage3.get("deployment_palette_rgb"):
        stage1["paint"]["palette_rgb"] = stage3["deployment_palette_rgb"]
    stage3["palette_rgb"] = stage1["paint"]["palette_rgb"]
    stage3["color_subset"] = stage2["palette_colors"]
    checkpoint_path = ROOT / stage3["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Stage-2 checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_palette_contract(stage1, stage2, stage3, checkpoint)
    model = ImpastoUNet(channels=tuple(stage2["unet_channels"]),
                        action_embedding_channels=int(stage2["action_embedding_channels"]),
                        dropout=float(stage2["dropout_probability"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return stage3, stage2, stage1, model, device, checkpoint


def canvas_gray(env):
    return srgb_gray(env.canvas.image())


def paper_weights(current, goal, config):
    p = config["planner"]
    changed = np.abs(np.asarray(goal) - np.asarray(current)) > float(p["objective_threshold"])
    kernel = np.ones((int(p["objective_dilation_kernel"]),
                      int(p["objective_dilation_kernel"])), np.uint8)
    dilated = cv2.dilate(changed.astype(np.uint8), kernel).astype(np.float32)
    weights = 1.0 + float(p["objective_changed_weight"]) * dilated
    return weights / max(float(weights.sum()), 1e-8), changed


def metrics(current_gray, actual_rgb, target_rgb, target_gray, initial_gray, config,
            lpips_model=None, device=None):
    weights, changed = paper_weights(initial_gray, target_gray, config)
    error = np.abs(np.asarray(current_gray, dtype=np.float32) - target_gray)
    result = {
        "unweighted_gray_l1": float(error.mean()),
        "weighted_gray_l1": float(np.sum(weights * error)),
        "changed_region_gray_l1": float(error[changed].mean()) if changed.any() else 0.0,
        "background_gray_l1": float(error[~changed].mean()) if (~changed).any() else 0.0,
        "changed_pixel_fraction": float(changed.mean()),
    }
    if lpips_model is not None:
        actual_rgb = np.asarray(actual_rgb)
        if actual_rgb.ndim == 2:
            actual_rgb = np.repeat(actual_rgb[..., None], 3, axis=-1)
        actual_rgb = actual_rgb.astype(np.float32)
        if actual_rgb.max() > 1.0:
            actual_rgb /= 255.0
        target = np.asarray(target_rgb, dtype=np.float32)
        if target.max() > 1.0:
            target /= 255.0
        x = torch.from_numpy(actual_rgb.transpose(2, 0, 1)[None]).to(device) * 2 - 1
        y = torch.from_numpy(target.transpose(2, 0, 1)[None]).to(device) * 2 - 1
        with torch.no_grad():
            result["lpips_alex"] = float(lpips_model(x, y).mean().cpu())
    else:
        result["lpips_alex"] = None
    return result


def text_font(size=22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def append_video_frame(writer, env, target_rgb, title):
    actual = Image.fromarray(env.canvas.image()).convert("RGB").resize((480, 560))
    target = Image.fromarray(np.asarray(target_rgb, dtype=np.uint8)).convert("RGB").resize((480, 560))
    frame = Image.new("RGB", (960, 610), (246, 246, 243))
    frame.paste(target, (0, 38))
    frame.paste(actual, (480, 38))
    draw = ImageDraw.Draw(frame)
    draw.text((12, 8), title, fill=(25, 25, 25), font=text_font(18))
    draw.text((160, 570), "Target", fill=(25, 25, 25), font=text_font(16))
    draw.text((640, 570), "Simulated canvas", fill=(25, 25, 25), font=text_font(16))
    writer.append_data(np.asarray(frame))


def save_triptych(path, target_rgb, current_rgb, predicted_gray=None):
    n = 3 if predicted_gray is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    axes = np.atleast_1d(axes)
    axes[0].imshow(target_rgb)
    axes[0].set_title("Target")
    axes[1].imshow(current_rgb)
    axes[1].set_title("Executed simulation")
    if predicted_gray is not None:
        axes[2].imshow(predicted_gray, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("MPC predicted terminal")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cycle_diagnostic(path, target_rgb, observed_rgb, proposals, actions, colors,
                          predicted_gray, actual_rgb, actual_gray, changed_mask,
                          canvas_width, canvas_height, palette):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].imshow(target_rgb)
    axes[0].set_title("Checkpoint target")
    axes[1].imshow(observed_rgb)
    axes[1].set_title("Observed before planning")

    overlay = Image.fromarray(np.asarray(target_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    palette_array = np.asarray(palette, dtype=np.uint8)
    for index, proposal in enumerate(proposals):
        stroke = Stroke(*map(float, proposal.action), int(proposal.color))
        points = stroke.points(np.linspace(0.0, 1.0, 100))
        pixels = np.stack([
            (points[:, 0] / canvas_width + .5) * (target_rgb.shape[1] - 1),
            (.5 - points[:, 1] / canvas_height) * (target_rgb.shape[0] - 1),
        ], axis=-1)
        color = tuple(int(value) for value in palette_array[proposal.color])
        draw.line([tuple(point) for point in pixels], fill=color, width=2)
        draw.ellipse((*pixels[0] - 3, *pixels[0] + 3), fill=color)
    for action, color_id in zip(actions, colors):
        stroke = Stroke(*map(float, action), int(color_id))
        points = stroke.points(np.linspace(0.0, 1.0, 100))
        pixels = np.stack([
            (points[:, 0] / canvas_width + .5) * (target_rgb.shape[1] - 1),
            (.5 - points[:, 1] / canvas_height) * (target_rgb.shape[0] - 1),
        ], axis=-1)
        color = tuple(int(value) for value in palette_array[color_id])
        coords = [tuple(point) for point in pixels]
        draw.line(coords, fill=(255, 255, 255), width=5)
        draw.line(coords, fill=color, width=2)
    axes[2].imshow(overlay)
    axes[2].set_title("Seeds (solid) / optimized (outlined)")
    axes[3].imshow(predicted_gray, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Predicted after executed actions")
    axes[4].imshow(actual_rgb)
    axes[4].set_title("Executed after same actions")
    error = np.abs(np.asarray(actual_gray) - np.asarray(predicted_gray))
    axes[5].imshow(error, cmap="magma", vmin=0, vmax=.25)
    axes[5].contour(changed_mask, levels=[.5], colors=["cyan"], linewidths=.5)
    axes[5].set_title(f"Prediction error (mean {error.mean():.4f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_single(config, stage2, stage1, model, device, report_dir, lpips_model):
    experiment = config["experiments"]
    seed = int(experiment["single_stroke_seed"])
    target_stroke = Stroke(-.045, .005, .060, .012, .40, .80, color=1)
    target = render_single_target(target_stroke,
                                  config["canvas_width_m"], config["canvas_height_m"],
                                  config["canvas_resolution"], seed,
                                  paint_config=stage1.get("paint"))
    planner_cfg = copy.deepcopy(config)
    planner_cfg["planner"]["horizon"] = 1
    planner_cfg["seed"] = seed
    dynamics = LearnedCanvasDynamics(model, device, stage2, planner_cfg)
    planner = AdaptiveMPCPlanner(dynamics, planner_cfg)
    env = PaintingEnv({**stage1, "canvas_resolution": int(config["canvas_resolution"])},
                      seed=seed, render=False)
    started = time.perf_counter()
    target_path = report_dir / "single_target.png"
    Image.fromarray(target.image_rgb).save(target_path)
    try:
        initial = canvas_gray(env)
        plan = planner.plan(initial, target.image_gray,
                            color_images=(env.canvas.image(), target.image_rgb))
        action = plan.actions[0]
        color = plan.colors[0]
        prediction, _ = dynamics.rollout(initial, plan.actions, plan.colors, plan.opacities)
        stroke = Stroke(*map(float, action), int(color),
                        dynamics.effective_opacity(plan.opacities[0]))
        video = imageio.get_writer(report_dir / "single_stroke.mp4", fps=1.5,
                                   codec="libx264", macro_block_size=None)
        try:
            append_video_frame(video, env, target.image_rgb, "Single-stroke MPC | before")
            execution = env.execute_stroke(stroke)
            append_video_frame(video, env, target.image_rgb, "Single-stroke MPC | after")
        finally:
            video.close()
        final_rgb = env.canvas.image()
        final_gray = canvas_gray(env)
        result_metrics = metrics(final_gray, final_rgb, target.image_rgb, target.image_gray,
                                 initial, planner_cfg, lpips_model, device)
        save_triptych(report_dir / "single_result.png", target.image_rgb, final_rgb, prediction)
        return {
            "target_source": target.source,
            "target_action": target_stroke.to_dict(),
            "planned_action": stroke.to_dict(),
            "planning_initial_weighted_l1": plan.initial_cost,
            "planning_final_weighted_l1": plan.final_cost,
            "planning_weighted_gray_l1": plan.final_cost,
            "planning_trace": plan.iterations,
            "execution": execution,
            "execution_success": bool(execution.get("success", False)),
            "execution_force_rmse_mean_N": float(execution["force_rmse_N"]),
            "execution_trajectory_rmse_mean_m": float(execution["trajectory_rmse_m"]),
            "metrics": result_metrics,
            "wall_seconds": time.perf_counter() - started,
            "target_image": str((report_dir / "single_target.png").relative_to(ROOT)),
            "result_image": str((report_dir / "single_result.png").relative_to(ROOT)),
            "video": str((report_dir / "single_stroke.mp4").relative_to(ROOT)),
        }
    finally:
        env.close()


def run_artwork(name, config, stage2, stage1, model, device, report_dir, lpips_model):
    seed = int(config["experiments"][f"{name}_seed"])
    total_strokes = render_target_prefix(name, paint_config=stage1.get("paint")).stroke_count
    checkpoints = sorted(set([5, 10, 15, total_strokes]))
    target_counts = checkpoints
    # Fig. 8 exposes additional intermediate frames.  When the execution
    # cursor is exactly on one of those frames, use the published prefix as
    # the PCA-RANSAC base.  This decomposes the newly added strokes instead
    # of repeatedly decomposing all strokes already present in the canvas.
    published_counts = (2, 5, 7, 10, 12, 15, total_strokes)
    target_frames = {
        count: render_target_prefix(name, count, config["canvas_width_m"],
                                    config["canvas_height_m"],
                                    config["canvas_resolution"], seed,
                                    paint_config=stage1.get("paint"))
        for count in set(target_counts).union(published_counts)
    }
    final_target = target_frames[total_strokes]
    Image.fromarray(final_target.image_rgb).save(report_dir / f"{name}_target.png")
    env = PaintingEnv({**stage1, "canvas_resolution": int(config["canvas_resolution"])},
                      seed=seed, render=False)
    planner_cfg = copy.deepcopy(config)
    planner_cfg["seed"] = seed
    if config.get("task_color_subsets"):
        planner_cfg["active_color_subset"] = config["task_color_subsets"][name]
    dynamics = LearnedCanvasDynamics(model, device, stage2, planner_cfg)
    planner = AdaptiveMPCPlanner(dynamics, planner_cfg)
    writer = imageio.get_writer(report_dir / f"{name}_painting.mp4", fps=2,
                                codec="libx264", macro_block_size=None)
    records, cycles, executed = [], [], 0
    warm_actions = warm_colors = warm_opacities = None
    started = time.perf_counter()
    diagnostics = []
    try:
        append_video_frame(writer, env, final_target.image_rgb, f"{name.title()} | initial")
        for goal_count in checkpoints:
            warm_actions = warm_colors = warm_opacities = None
            goal = target_frames[goal_count].image_gray
            while executed < goal_count:
                # The terminal image of a prediction window must correspond to
                # its remaining actions, rather than fictitious future strokes.
                planner.pc["horizon"] = min(int(config["planner"]["horizon"]),
                                             goal_count - executed)
                observed = canvas_gray(env)
                observed_rgb = env.canvas.image()
                # Appendix C initializes from the actually observed base and
                # the current checkpoint goal.  A published prefix frame is
                # useful for visual diagnostics, but it is not the robot's
                # observation and must not replace it during normal planning.
                proposal_images = None
                if config["experiments"].get("use_published_prefix_targets", False):
                    # The public figure only exposes prefixes at 2/5/7/10/12/15.
                    # Replanning after the first action of a pair must still
                    # decompose the same incremental target window; falling
                    # back to the live, already-mismatched canvas would turn
                    # execution error into a new geometric proposal.
                    # A published checkpoint is a valid geometric base only
                    # when it represents the exact observed stroke count.
                    # Reusing an older frame after additional live execution
                    # would turn execution/model mismatch into a fake target
                    # residual and seed an incorrect long stroke.
                    prefix_count = executed if executed in published_counts else None
                    proposal_frame = (target_frames.get(prefix_count)
                                      if prefix_count is not None else None)
                    if proposal_frame is not None:
                        proposal_images = (proposal_frame.image_gray,
                                           proposal_frame.image_rgb)
                plan = planner.plan(
                    observed, goal,
                    color_images=(observed_rgb, target_frames[goal_count].image_rgb),
                    proposal_images=proposal_images,
                    initial=warm_actions, colors=warm_colors, opacities=warm_opacities)
                print(f"{name}: {executed}->{goal_count}, horizon {len(plan.actions)}, cost {plan.initial_cost:.5f}->{plan.final_cost:.5f}", flush=True)
                _, predicted_frames = dynamics.rollout(
                    observed, plan.actions, plan.colors, plan.opacities)
                cycles.append({"goal_stroke_count": goal_count,
                               "executed_before_replan": executed,
                               "planning_initial_weighted_l1": plan.initial_cost,
                               "planning_final_weighted_l1": plan.final_cost,
                               "planning_trace": plan.iterations,
                               "proposals": len(plan.proposals),
                               "horizon": len(plan.actions)})
                count = min(int(config["planner"]["execute_steps"]), goal_count - executed)
                predicted_after_actions = predicted_frames[count - 1]
                for i in range(count):
                    stroke = Stroke(*map(float, plan.actions[i]), int(plan.colors[i]),
                                    dynamics.effective_opacity(plan.opacities[i]))
                    result = env.execute_stroke(stroke)
                    if not result.get("success"):
                        raise RuntimeError(f"{name} planned stroke failed at {executed}: {result}")
                    executed += 1
                    records.append({"step": executed, "stroke": stroke.to_dict(),
                                    "execution": result,
                                    "goal_count": goal_count,
                                    "planner_predicted_cost": plan.final_cost})
                    append_video_frame(writer, env, final_target.image_rgb,
                                       f"{name.title()} | executed {executed}/{total_strokes} | target {goal_count}")
                if config["planner"].get("warm_start", False):
                    warm_actions = plan.actions[count:].copy()
                    warm_colors = plan.colors[count:]
                    warm_opacities = plan.opacities[count:]
                actual_after_rgb = env.canvas.image()
                actual_after_gray = canvas_gray(env)
                _, objective_mask = paper_weights(observed, goal, planner_cfg)
                prediction_error = np.abs(predicted_after_actions - actual_after_gray)
                cycle_diagnostic = report_dir / f"{name}_cycle_{len(cycles):02d}.png"
                save_cycle_diagnostic(
                    cycle_diagnostic, target_frames[goal_count].image_rgb, observed_rgb,
                    plan.proposals, plan.actions[:count], plan.colors[:count],
                    predicted_after_actions, actual_after_rgb, actual_after_gray,
                    objective_mask, config["canvas_width_m"], config["canvas_height_m"],
                    config["palette_rgb"])
                np.savez_compressed(report_dir / f"{name}_cycle_{len(cycles):02d}.npz",
                    before_rgb=observed_rgb, target_rgb=target_frames[goal_count].image_rgb,
                    after_rgb=actual_after_rgb, predicted=predicted_after_actions,
                    predicted_terminal=plan.predicted, actions=plan.actions, colors=plan.colors,
                    opacities=plan.opacities, mask=objective_mask,
                    seeds=plan.initial_actions, seed_colors=plan.initial_colors,
                    initialized_from_warm_start=warm_actions is not None,
                    seed_masks=np.array([p.mask for p in plan.proposals]))
                Image.fromarray(actual_after_rgb).save(report_dir / f"{name}_after_{executed:02d}.png")
                cycle_metrics = {
                    "executed_actions": count,
                    "before_goal_weighted_l1": planner.cost(observed, goal, planner.objective_weights(observed, goal)),
                    "after_goal_weighted_l1": planner.cost(actual_after_gray, goal, planner.objective_weights(observed, goal)),
                    "prediction_vs_execution_l1": float(prediction_error.mean()),
                    "prediction_vs_execution_goal_region_l1": (
                        float(prediction_error[objective_mask].mean()) if objective_mask.any() else 0.0),
                    "predicted_after_actions_goal_weighted_l1": float(np.sum(
                        paper_weights(observed, goal, planner_cfg)[0]
                        * np.abs(predicted_after_actions - goal))),
                    "diagnostic_image": cycle_diagnostic.name,
                }
                cycles[-1]["prediction_execution"] = cycle_metrics
                diagnostics.append(str(cycle_diagnostic.relative_to(ROOT)))
        actual_rgb = env.canvas.image()
        actual_gray = canvas_gray(env)
        result_metrics = metrics(actual_gray, actual_rgb, final_target.image_rgb, final_target.image_gray,
                                 srgb_gray(Canvas(config["canvas_width_m"], config["canvas_height_m"],
                                                 config["canvas_resolution"], seed).image()),
                                 planner_cfg, lpips_model, device)
        save_triptych(report_dir / f"{name}_result.png", final_target.image_rgb, actual_rgb)
        return {
            "target_source": final_target.source,
            "target_strokes": total_strokes,
            "executed_strokes": executed,
            "replanning_cycles": len(cycles),
            "cycles": cycles,
            "prediction_execution_l1_mean": float(np.mean([
                cycle["prediction_execution"]["prediction_vs_execution_l1"]
                for cycle in cycles])),
            "prediction_execution_goal_region_l1_mean": float(np.mean([
                cycle["prediction_execution"]["prediction_vs_execution_goal_region_l1"]
                for cycle in cycles])),
            "mean_planned_opacity": float(np.mean([
                record["stroke"]["opacity"] for record in records])),
            "cycle_diagnostics": diagnostics,
            "stroke_records": records,
            "planning_weighted_gray_l1": float(np.mean(
                [cycle["planning_final_weighted_l1"] for cycle in cycles])),
            "mean_planning_objective": float(np.mean(
                [cycle["planning_final_weighted_l1"] for cycle in cycles])),
            "execution_force_rmse_mean_N": float(np.mean(
                [record["execution"]["force_rmse_N"] for record in records])),
            "execution_trajectory_rmse_mean_m": float(np.mean(
                [record["execution"]["trajectory_rmse_m"] for record in records])),
            "metrics": result_metrics,
            "wall_seconds": time.perf_counter() - started,
            "target_image": str((report_dir / f"{name}_target.png").relative_to(ROOT)),
            "result_image": str((report_dir / f"{name}_result.png").relative_to(ROOT)),
            "video": str((report_dir / f"{name}_painting.mp4").relative_to(ROOT)),
        }
    finally:
        writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed experiments only when resolved settings match")
    parser.add_argument("--report-dir", help="Write this experiment into a separate report directory")
    parser.add_argument("--overpaint-weight", type=float,
                        help="Penalize predicted paint darker than the goal in the planning objective")
    args = parser.parse_args()
    config, stage2, stage1, model, device, checkpoint = load_system()
    if args.report_dir is not None:
        config["report_dir"] = args.report_dir
    if args.overpaint_weight is not None:
        config["planner"]["overpaint_weight"] = args.overpaint_weight
    if float(config["planner"].get("overpaint_weight", 1.0)) < 1.0:
        parser.error("--overpaint-weight must be at least 1")
    report_dir = ROOT / config["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = None
    lpips_error = None
    if os.environ.get("IMPASTO_ENABLE_LPIPS", "0") == "1":
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
        except Exception as exc:
            lpips_error = f"{type(exc).__name__}: {exc}"
    else:
        lpips_error = ("disabled by default; set IMPASTO_ENABLE_LPIPS=1 after verifying AlexNet weights; "
                       "a previous download did not pass SHA-256 validation")
    stage2_metrics_path = ROOT / stage2.get("report_dir", "reports/stage2") / "training_metrics.json"
    stage2_metrics = json.loads(stage2_metrics_path.read_text()) if stage2_metrics_path.is_file() else {}
    checkpoint_sha256 = hashlib.sha256((ROOT / config["checkpoint"]).read_bytes()).hexdigest()
    completed = {}
    if args.resume and (report_dir / "metrics.json").is_file():
        previous_config = json.loads((report_dir / "resolved_config.json").read_text())
        def relevant_settings(value):
            value = copy.deepcopy(value)
            # Enabling another experiment cannot affect a completed experiment.
            for key in list(value["experiments"]):
                if key.startswith("run_"):
                    del value["experiments"][key]
            return value
        if relevant_settings(config) != relevant_settings(previous_config):
            raise ValueError("Resume refused: resolved experiment settings changed")
        previous = json.loads((report_dir / "metrics.json").read_text())
        if previous["paper_protocol"]["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Resume refused: checkpoint changed")
        completed = previous["experiments"]
    (report_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    results = {
        "paper_protocol": {
            "mpc_horizon": int(config["planner"]["horizon"]),
            "execute_steps_before_replan": int(config["planner"]["execute_steps"]),
            "baselines": "not reproduced, per project scope",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_matches_stage2_report": checkpoint_sha256 == stage2_metrics.get("best_checkpoint_sha256"),
            "target_scope": "public Fig. 8 panel crops; original high-resolution artist files and action annotations unavailable",
            "unpublished_planner_parameters": f"local choices in {config['config_file']}",
            "planning_objective": "asymmetric weighted gray L1 + RGB auxiliary weight * weighted RGB L1",
            "overpaint_weight": config["planner"].get("overpaint_weight", 1.0),
            "rgb_objective_weight": config["planner"].get("rgb_objective_weight", 0),
        },
        "device": str(device),
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error or (
            "not run; the attempted torchvision AlexNet weight download failed SHA-256 validation"),
        "experiments": completed,
    }
    if config["experiments"].get("run_single_stroke", True) and "single_stroke" not in completed:
        results["experiments"]["single_stroke"] = run_single(
            config, stage2, stage1, model, device, report_dir, lpips_model)
        print(json.dumps({"single_stroke": results["experiments"]["single_stroke"]["metrics"]}), flush=True)
    for name in ("flower", "fish"):
        if config["experiments"].get(f"run_{name}", True) and name not in completed:
            results["experiments"][name] = run_artwork(
                name, config, stage2, stage1, model, device, report_dir, lpips_model)
            print(json.dumps({name: results["experiments"][name]["metrics"],
                              "replanning_cycles": results["experiments"][name]["replanning_cycles"]}), flush=True)
            (report_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=True))
    build_report(config, results, report_dir)
    print(json.dumps({"report": str((report_dir / 'report.md').relative_to(ROOT)),
                      "experiments": {name: item["metrics"] for name, item in results["experiments"].items()}}, indent=2))


def build_report(config, results, report_dir):
    rows = []
    for name, item in results["experiments"].items():
        metric = item["metrics"]
        rows.append(f"|{name}|{item.get('target_strokes', 1)}|{item.get('executed_strokes', 1)}|"
                    f"{item.get('mean_planning_objective', item['planning_weighted_gray_l1']):.5f}|{metric['weighted_gray_l1']:.5f}|"
                    f"{metric['lpips_alex'] if metric['lpips_alex'] is not None else 'unavailable'}|")
    media = []
    for name, item in results["experiments"].items():
        media.append(f"### {name}\n\n![{name} result]({Path(item['result_image']).name})\n\n"
                     f"[Target]({Path(item['target_image']).name}) · [Execution video]({Path(item['video']).name})\n")
    quality_rows = []
    for name, item in results["experiments"].items():
        quality_rows.append(
            f"|{name}|{int(item.get('execution_success', all(r['execution']['success'] for r in item.get('stroke_records', []))))}|"
            f"{item['execution_force_rmse_mean_N']:.5f}|{item['execution_trajectory_rmse_mean_m']:.6f}|")
    prediction_rows = []
    for name, item in results["experiments"].items():
        if "prediction_execution_l1_mean" in item:
            prediction_rows.append(
                f"|{name}|{item['prediction_execution_l1_mean']:.5f}|"
                f"{item['prediction_execution_goal_region_l1_mean']:.5f}|"
                f"{item['mean_planned_opacity']:.3f}|")
    diagnostics = []
    for name, item in results["experiments"].items():
        if item.get("cycle_diagnostics"):
            diagnostics.append(f"### {name}\n\n" + "\n\n".join(
                f"![{Path(path).stem}]({Path(path).name})" for path in item["cycle_diagnostics"]))
    lpips_line = ("LPIPS AlexNet is enabled with the pretrained feature backbone."
                  if results["lpips_available"] else
                  f"LPIPS could not be initialized: `{results['lpips_error']}`")
    text = f"""# IMPASTO 阶段三：学习模型驱动的笔画规划

## 结果

规划器使用阶段二训练的 IMPASTO-UNet、PCA–RANSAC 初始化、MPPI elite softmax 序列更新及 CMA 风格自适应。多笔绘画采用 H={config['planner']['horizon']}，每轮最多执行 {config['planner']['execute_steps']} 笔后读取仿真画布并重新规划。本配置的规划目标为加权灰度 L1（过度上色系数 {config['planner'].get('overpaint_weight', 1.0)}）+ {config['planner'].get('rgb_objective_weight', 0)} × RGB 辅助损失。这两项加权修改属于本地实验；不将综合规划代价标作论文的灰度 L1。

|任务|目标笔数|执行笔数|综合规划代价均值|执行加权灰度 L1|LPIPS AlexNet|
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

规划代价是各重规划周期终端预测代价的均值；执行 L1 是最终真实仿真画布与目标图的误差，两列含义不同。兼容旧日志而保留的 `planning_weighted_gray_l1` 字段在启用 RGB 辅助项时也是综合代价。完整全画布 L1、改变区域 L1、逐周期代价、每笔力控及轨迹数据见 `metrics.json`。本阶段没有运行或比较任何 baseline。{lpips_line}

## 执行质量

|任务|全部执行成功|法向力 RMSE (N)|轨迹 RMSE (m)|
|---|---:|---:|---:|
{chr(10).join(quality_rows)}

## 模型预测与实际执行

|任务|执行相同动作后的全画布预测 L1|目标差异区预测 L1|平均透明度 α|
|---|---:|---:|---:|
{chr(10).join(prediction_rows)}

此处逐轮比较模型对已执行动作的预测与同一轮真实仿真观测。

## 任务结果

{chr(10).join(media)}

## 论文对齐与边界

- 单笔：H=1，以目标差分初始化并优化笔画参数，再执行该动作并读取真实仿真画布。当前 `single_stroke` 是本地生成的诊断曲线，不能代替论文五位画家的人类笔迹实验。
- Flower/Fish：保留论文所述 17/18 笔目标规模；目标图和 2/5/7/10/12/15 笔公开检查点来自作者 Fig. 8，使用目标笔数 5、10、15 和最终状态作为窗口目标，H=5，每轮执行两笔并闭环重规划。
- 论文使用的 H=5、每轮两笔及 PCA–RANSAC/MPPI/CMA 规划结构已落实。论文未公开候选数、迭代数、softmax 温度、RANSAC 阈值和 CMA 系数；本地取值见 `{config.get("config_file", "configs/stage3.json")}`。
- 论文没有随文章发布 artist Flower/Fish 的原始高分辨率文件和逐笔颜色/动作。这里使用作者公开 Fig. 8 的低分辨率目标 panel，保留目标笔画数并由规划器推断动作，不冒充原始逐笔标注或定量复刻。
    - 修订配置使用仿真执行采集的 25×25 颜料块均值、协方差和 Mahalanobis 分类，几何改变后重新取色。每色两种力，协方差加正则项；这是本地数据库，不是作者的实物数据库。中心带采样近似论文中心圆盘与预测足迹交集。
    - 历史配置曾把附录 B 的分类比值作为连续透明度，并乘本地系数；该做法与真实沉积不等价，修订配置仅使用不透明分支。当前 `opacity_scale={config.get('opacity_scale', 1.0):.3f}`；不能据此声称实现了论文完整透明度模块。
- 本地阶段一使用 MuJoCo penalty contact 与简化颜料沉积，故执行结果是仿真验证，不等价于真实油画物理或论文硬件试验。
- LPIPS AlexNet 未计算：torchvision 预训练 AlexNet 权重下载未通过 SHA-256 校验；不会以随机网络给出感知指标。

## 可复现文件

- 规划器配置：`{config.get("config_file", "configs/stage3.json")}`
- 聚合数值和逐笔动作/力控日志：`{config["report_dir"]}/metrics.json`
- Stage-2 checkpoint SHA-256：`{results['paper_protocol']['checkpoint_sha256']}`
- 运行命令：`IMPASTO_STAGE3_CONFIG={config.get("config_file", "configs/stage3.json")} bash scripts/run.sh scripts/run_stage3.py --report-dir {config['report_dir']} --overpaint-weight {config['planner'].get('overpaint_weight', 1.0)}`

## 逐轮诊断图

{chr(10).join(diagnostics)}
"""
    (report_dir / "report.md").write_text(text)


if __name__ == "__main__":
    main()
