#!/usr/bin/env python3
"""Replay saved Stage-3 actions to film continuous MuJoCo execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/stage3_refined")
    parser.add_argument("--task", choices=("flower", "fish"), default="flower")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    report = ROOT / args.report_dir
    config = json.loads((report / "resolved_config.json").read_text())
    metrics = json.loads((report / "metrics.json").read_text())
    stage1 = config.get("resolved_stage1")
    if stage1 is None:
        stage1 = json.loads((ROOT / config["stage1_config"]).read_text())
    stage1["canvas_resolution"] = int(config["canvas_resolution"])
    experiment = metrics["experiments"][args.task]
    target = Image.open(report / f"{args.task}_target.png").convert("RGB").resize((160, 180))
    env = PaintingEnv(stage1, seed=int(config["experiments"][f"{args.task}_seed"]), render=True)
    video_path = report / f"{args.task}_scene_replay.mp4"
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264",
                                macro_block_size=None)
    next_time = 0.0
    frames = 0

    def hook(scene, obs, force=False):
        nonlocal next_time, frames
        if not force and obs.time + 1e-9 < next_time:
            return
        rgb = scene.render("overview_camera", width=640, height=360)
        frame = Image.new("RGB", (960, 360), (245, 245, 242))
        frame.paste(Image.fromarray(rgb), (0, 0))
        frame.paste(target, (640, 24))
        current = Image.fromarray(scene.canvas.image()).convert("RGB").resize((160, 180))
        frame.paste(current, (800, 24))
        draw = ImageDraw.Draw(frame)
        draw.text((648, 4), "Target", fill=(20, 20, 20))
        draw.text((808, 4), "Live canvas", fill=(20, 20, 20))
        draw.text((650, 225), f"{args.task} | {scene.phase}", fill=(20, 20, 20))
        draw.text((650, 250), f"t={obs.time:.1f}s  F={obs.normal_force:.2f}N",
                  fill=(20, 20, 20))
        writer.append_data(np.asarray(frame))
        frames += 1
        next_time = (np.floor((obs.time + 1e-9) * args.fps) + 1) / args.fps

    try:
        hook(env, env.observe(np.array([np.nan, np.nan, np.nan])), force=True)
        for record in experiment["stroke_records"]:
            stroke = Stroke(**record["stroke"])
            result = env.execute_stroke(stroke, render_hook=hook)
            if not result["success"]:
                raise RuntimeError(f"Replay failed at stroke {record['step']}: {result}")
            print(f"Replayed {record['step']}/{experiment['executed_strokes']}", flush=True)
        hook(env, env.observe(np.array([np.nan, np.nan, np.nan])), force=True)
        reference = np.asarray(Image.open(
            report / f"{args.task}_after_{experiment['executed_strokes']:02d}.png"))
        difference = np.abs(env.canvas.image().astype(np.int16) - reference.astype(np.int16))
        result = {"task": args.task, "frames": frames, "fps": args.fps,
                  "video": str(video_path.relative_to(ROOT)),
                  "final_canvas_max_abs_rgb_difference": int(difference.max()),
                  "final_canvas_mean_abs_rgb_difference": float(difference.mean())}
        (report / f"{args.task}_scene_replay.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if difference.max() > 0:
            raise AssertionError("Replay final canvas does not match the planned run")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
