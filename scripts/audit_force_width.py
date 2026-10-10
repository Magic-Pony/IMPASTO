#!/usr/bin/env python3
"""Replay logged Stage-3 strokes while varying only their normal force."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from impasto.actions import Stroke
from impasto.sim.paint import BrushState, Canvas
from audit_refinement import region_metrics

ROOT = Path(__file__).resolve().parents[1]


def replay(records, stage3_config, stage1_config, seed, force_scale=1.0):
    canvas = Canvas(stage3_config["canvas_width_m"], stage3_config["canvas_height_m"],
                    stage3_config["canvas_resolution"], seed=seed,
                    paint_config=stage1_config["paint"])
    brush = BrushState()
    dt = float(stage1_config["timestep"]) * int(stage1_config.get("control_decimation", 2))
    speed = float(stage1_config.get("stroke_speed", .045))
    reload_each = bool(stage1_config.get("reload_brush_each_stroke", False))
    for record in records:
        item = record["stroke"]
        force = max(.4, float(item["force"]) * force_scale)
        stroke = Stroke(item["x"], item["y"], item["length"], item["bend"],
                        item["angle"], force, int(item["color"]),
                        float(item.get("opacity", 1.0)))
        if (brush.color != stroke.color or brush.load < .08 or reload_each):
            if brush.color >= 0:
                brush.clean()
            brush.dip(stroke.color)
        brush.opacity = stroke.opacity
        canvas.reset_path()
        duration = stroke.arc_length() / speed
        count = max(2, int(np.ceil(duration / dt)))
        for s in np.linspace(0.0, 1.0, count + 1)[1:]:
            canvas.update(stroke.points(s), stroke.force, brush, dt, contacting=True)
        canvas.reset_path()
    return canvas.image()


def audit_task(name, metrics, stage3_config, stage1_config, report_dir, scales, out_dir):
    experiment = metrics["experiments"][name]
    target = np.asarray(Image.open(report_dir / f"{name}_target.png").convert("RGB"))
    executed = int(experiment["executed_strokes"])
    logged = np.asarray(Image.open(
        report_dir / f"{name}_after_{executed:02d}.png").convert("RGB"))
    seed = int(stage3_config["experiments"][f"{name}_seed"])
    records = experiment["stroke_records"]
    renders = {scale: replay(records, stage3_config, stage1_config, seed, scale)
               for scale in scales}
    actual = replay(records, stage3_config, stage1_config, seed, 1.0)
    reconstruction_error = float(np.mean(np.abs(actual.astype(float)-logged.astype(float))) / 255)

    cols = [("Target", target), ("Logged result", logged),
            ("Replay, original force", actual)]
    cols.extend((f"Force x{scale:.2f}", renders[scale])
                for scale in scales if scale < 1.0)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.0*len(cols), 4.8))
    for ax, (title, image) in zip(np.atleast_1d(axes), cols):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_force_width.png", dpi=150)
    plt.close(fig)

    baseline = region_metrics(actual, target)
    rows = {"baseline_force_scale": 1.0,
            "reconstruction_mean_abs_rgb_error": reconstruction_error,
            "metrics": baseline}
    for scale, image in renders.items():
        rows[f"force_scale_{scale:.2f}"] = {
            "metrics": region_metrics(image, target),
            "mean_abs_rgb_delta_from_baseline": float(
                np.mean(np.abs(image.astype(float)-actual.astype(float))) / 255),
        }
        Image.fromarray(image).save(out_dir / f"{name}_force_x{scale:.2f}.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default="configs/stage3_refined.json")
    parser.add_argument("--stage1-config", default="configs/stage1_refined.json")
    parser.add_argument("--report-dir", default="reports/stage3_soft_edge")
    parser.add_argument("--out-dir", default="reports/refinement_soft_edge/force_width")
    args = parser.parse_args()
    stage3 = json.loads((ROOT / args.stage3_config).read_text())
    stage1 = json.loads((ROOT / args.stage1_config).read_text())
    report_dir, out_dir = ROOT / args.report_dir, ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((report_dir / "metrics.json").read_text())
    result = {name: audit_task(name, metrics, stage3, stage1, report_dir,
                               [1.0, .9, .75, .6], out_dir)
              for name in ("flower", "fish")}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
