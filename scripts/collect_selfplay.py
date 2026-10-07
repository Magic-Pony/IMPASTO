#!/usr/bin/env python3
"""Collect the Stage-2 robot self-play transition dataset."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import multiprocessing as mp
import time

import numpy as np

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.stage2.raster import (crop_canvas_gray, crop_for_stroke,
                                    normalize_patch_action, patch_action,
                                    render_stroke_patch)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stage2"
QUALITY_FIELDS = (
    "target_force_N", "force_mean_N", "force_rmse_N", "contact_fraction",
    "trajectory_rmse_m", "brush_tilt_mean_deg", "force_acquisition_duration_s",
    "failed_attempt_count", "stroke_wall_seconds",
)


def load_config():
    stage2 = json.loads((ROOT / "configs/stage2.json").read_text())
    stage1 = json.loads((ROOT / stage2["stage1_config"]).read_text())
    return stage2, stage1


def draw_stroke(rng, cfg, color):
    b = cfg["action_bounds"]
    for _ in range(1000):
        stroke = Stroke(
            float(rng.uniform(*b["x_m"])), float(rng.uniform(*b["y_m"])),
            float(rng.uniform(*b["length_m"])), float(rng.uniform(*b["bend_m"])),
            float(rng.uniform(*b["angle_rad"])), float(rng.uniform(*b["force_N"])),
            int(color),
        )
        points = stroke.points(np.linspace(0, 1, 64))
        if (np.abs(points[:, 0]).max() <= cfg["workspace_abs_limit_m"][0]
                and np.abs(points[:, 1]).max() <= cfg["workspace_abs_limit_m"][1]):
            return stroke
    raise RuntimeError("Could not sample a reachable Stage-2 stroke")


def collect_episode(task):
    episode_id, seed, stage2_cfg, stage1_cfg, output_path = task
    rng = np.random.default_rng(seed)
    env = PaintingEnv(stage1_cfg, seed=seed, render=False)
    images, targets, actions, raw_actions = [], [], [], []
    world_actions, crop_transforms, colors, quality, metrics = [], [], [], [], []
    started = time.perf_counter()
    color_set = stage2_cfg["palette_colors"]
    try:
        for step in range(int(stage2_cfg["samples_per_episode"])):
            color = int(color_set[(step // int(stage2_cfg["color_block_size"])) % len(color_set)])
            failed_attempts = []
            for attempt in range(int(stage2_cfg["max_attempts_per_sample"])):
                stroke = draw_stroke(rng, stage2_cfg, color)
                crop = crop_for_stroke(stroke, int(stage2_cfg["patch_size"]), float(stage2_cfg["crop_margin_m"]))
                before = env.canvas.image().copy()
                current = crop_canvas_gray(before, crop, env.canvas.width, env.canvas.height)
                rendered = render_stroke_patch(stroke, crop, stage2_cfg["force_radius_m"])
                local_action = patch_action(stroke, crop)
                normalized = normalize_patch_action(local_action,
                                                    stage2_cfg["action_bounds"]["force_N"],
                                                    int(stage2_cfg["patch_size"]))
                result = env.execute_stroke(stroke)
                valid = (result.get("success", False)
                         and result.get("contact_fraction", 0.0) >= .99
                         and result.get("force_acquired", False))
                if valid:
                    break
                failed_attempts.append({"attempt": attempt, "reason": result.get("reason"),
                                        "contact_fraction": result.get("contact_fraction"),
                                        "force_acquired": result.get("force_acquired")})
            else:
                raise RuntimeError(f"episode {episode_id} step {step} exhausted attempts: {failed_attempts}")
            after = crop_canvas_gray(env.canvas.image(), crop, env.canvas.width, env.canvas.height)
            images.append(np.stack([current, rendered], axis=0))
            targets.append(after[None])
            actions.append(normalized)
            raw_actions.append(local_action)
            world_actions.append([stroke.x, stroke.y, stroke.length, stroke.bend, stroke.angle, stroke.force])
            crop_transforms.append([crop.center_x_m, crop.center_y_m, crop.side_m])
            colors.append(color)
            quality.append([
                result.get("target_force_N", stroke.force),
                result.get("force_mean_N", 0.0),
                result.get("force_rmse_N", 0.0),
                result.get("contact_fraction", 0.0),
                result.get("trajectory_rmse_m", float("nan")),
                result.get("brush_tilt_mean_deg", float("nan")),
                result.get("force_acquisition_duration_s", float("nan")),
                len(failed_attempts),
                result.get("wall_seconds", float("nan")),
            ])
            metrics.append({"step": step, "success": True, "target_force_N": stroke.force,
                            "force_mean_N": result.get("force_mean_N", 0.0),
                            "force_rmse_N": result.get("force_rmse_N", 0.0),
                            "contact_fraction": result.get("contact_fraction", 0.0),
                            "failed_attempts": failed_attempts,
                            "sim_time_s": float(env.data.time),
                            "force_acquisition_s": result.get("force_acquisition_duration_s")})
        payload = {
            "episode_id": episode_id, "seed": seed, "samples": len(images),
            "wall_seconds": time.perf_counter() - started,
            "simulation_seconds": float(env.data.time), "metrics": metrics,
        }
        tmp_path = Path(f"{output_path}.tmp")
        with tmp_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                images=np.uint8(np.clip(np.rint(np.asarray(images) * 255), 0, 255)),
                targets=np.uint8(np.clip(np.rint(np.asarray(targets) * 255), 0, 255)),
                actions=np.asarray(actions, dtype=np.float32),
                raw_actions=np.asarray(raw_actions, dtype=np.float32),
                world_actions=np.asarray(world_actions, dtype=np.float32),
                crop_transforms=np.asarray(crop_transforms, dtype=np.float32),
                colors=np.asarray(colors, dtype=np.int64),
                quality=np.asarray(quality, dtype=np.float32),
                episode_ids=np.full(len(images), episode_id, dtype=np.int64),
                step_ids=np.arange(len(images), dtype=np.int64),
            )
        os.replace(tmp_path, output_path)
        return payload
    finally:
        env.close()


def combine_episodes(paths, output):
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    keys = ("images", "targets", "actions", "raw_actions", "world_actions",
            "crop_transforms", "colors", "quality", "episode_ids", "step_ids")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    tmp_path = Path(f"{output}.tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp_path, output)


def validate_collection(results, episode_paths, combined_path, cfg):
    expected_ids = list(range(int(cfg["episodes"])))
    result_ids = [int(item["episode_id"]) for item in results]
    if result_ids != expected_ids:
        raise RuntimeError(f"Episode results do not match configured IDs: {result_ids}")
    expected_per_episode = int(cfg["samples_per_episode"])
    for result, path in zip(results, episode_paths):
        if int(result["samples"]) != expected_per_episode or not path.is_file():
            raise RuntimeError(f"Incomplete episode {result['episode_id']}: {result}")
        with np.load(path, allow_pickle=False) as data:
            lengths = {len(data[key]) for key in ("images", "targets", "actions", "quality",
                                                   "episode_ids", "step_ids")}
            if lengths != {expected_per_episode}:
                raise RuntimeError(f"Episode {result['episode_id']} has inconsistent array lengths")
            if not np.isfinite(data["quality"]).all():
                raise RuntimeError(f"Episode {result['episode_id']} has non-finite quality metrics")
    train_ids = set(map(int, cfg["split"]["train_episode_ids"]))
    validation_ids = set(map(int, cfg["split"]["validation_episode_ids"]))
    if train_ids & validation_ids or train_ids | validation_ids != set(expected_ids):
        raise RuntimeError("Train and validation episode IDs must be disjoint and cover the dataset")
    train_count = sum(int(item["samples"]) for item in results
                      if int(item["episode_id"]) in train_ids)
    validation_count = sum(int(item["samples"]) for item in results
                           if int(item["episode_id"]) in validation_ids)
    if train_count != int(cfg["train_samples"]) or validation_count != int(cfg["validation_samples"]):
        raise RuntimeError(f"Expected configured train/validation counts, got {train_count}/{validation_count}")
    with np.load(combined_path, allow_pickle=False) as data:
        total = int(cfg["train_samples"]) + int(cfg["validation_samples"])
        if len(data["images"]) != total or len(data["quality"]) != total:
            raise RuntimeError(f"Combined archive has {len(data['images'])} samples, expected {total}")
        if not np.isfinite(data["quality"]).all():
            raise RuntimeError("Combined archive contains non-finite quality metrics")
        for episode_id in expected_ids:
            ids = data["episode_ids"] == episode_id
            if int(ids.sum()) != expected_per_episode:
                raise RuntimeError(f"Combined archive has wrong sample count for episode {episode_id}")
            if not np.array_equal(data["step_ids"][ids], np.arange(expected_per_episode)):
                raise RuntimeError(f"Combined archive has invalid step order for episode {episode_id}")
    return train_count, validation_count


def main():
    stage2_cfg, stage1_cfg = load_config()
    data_dir = ROOT / stage2_cfg["dataset_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for episode_id in range(int(stage2_cfg["episodes"])):
        tasks.append((episode_id, int(stage2_cfg["seed"]) + episode_id,
                      stage2_cfg, stage1_cfg, str(data_dir / f"episode_{episode_id:02d}.npz")))
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(stage2_cfg["collection_workers"]), mp_context=context) as pool:
        results = list(pool.map(collect_episode, tasks))
    results.sort(key=lambda item: item["episode_id"])
    combined = data_dir / "stage2_selfplay.npz"
    episode_paths = [Path(task[-1]) for task in tasks]
    combine_episodes(episode_paths, combined)
    train_count, validation_count = validate_collection(results, episode_paths, combined, stage2_cfg)
    total = sum(item["samples"] for item in results)
    manifest = {
        "dataset": "IMPASTO Stage-2 self-play transitions",
        "paper_training_samples": int(stage2_cfg["train_samples"]),
        "paper_validation_samples": int(stage2_cfg["validation_samples"]),
        "samples": total, "episodes": results,
        "quality_fields": list(QUALITY_FIELDS),
        "train_samples_verified": train_count,
        "validation_samples_verified": validation_count,
        "train_episode_ids": stage2_cfg["split"]["train_episode_ids"],
        "validation_episode_ids": stage2_cfg["split"]["validation_episode_ids"],
        "config": stage2_cfg, "stage1_config": stage1_cfg,
        "wall_seconds": time.perf_counter() - started,
        "combined_path": str(combined.relative_to(ROOT)),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True))
    (REPORT / "collection_metrics.json").write_text(json.dumps(manifest, indent=2, allow_nan=True))
    print(json.dumps({"samples": total, "wall_seconds": manifest["wall_seconds"],
                      "dataset": str(combined)}, indent=2))


if __name__ == "__main__":
    main()
