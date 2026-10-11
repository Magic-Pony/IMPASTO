#!/usr/bin/env python3
"""Audit force stability and model error on artwork-independent brushwork."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.stage2.raster import srgb_gray
from impasto.stage3.planner import LearnedCanvasDynamics
from scripts.run_stage3 import load_system

ROOT = Path(__file__).resolve().parents[1]


def evaluate_case(stage1, stage3, dynamics, stroke, seed, overlap=False):
    env = PaintingEnv(stage1, seed=seed, render=False)
    try:
        if overlap:
            result = env.execute_stroke(Stroke(-.055, -.008, .055, -.014, 1.0, .95, 2))
            if not result["success"]:
                raise RuntimeError(result)
        before = srgb_gray(env.canvas.image())
        action = np.array([stroke.x, stroke.y, stroke.length, stroke.bend,
                           stroke.angle, stroke.force], np.float32)
        pred, local, crop, _ = dynamics.step(before, action, stroke.color)
        execution = env.execute_stroke(stroke)
        if not execution["success"]:
            raise RuntimeError(execution)
        actual = srgb_gray(env.canvas.image())
        # Union of changed pixels, relative to the same pre-stroke canvas.
        # This prevents empty background from hiding thin-stroke errors.
        union = (abs(actual-before) > .015) | (abs(pred-before) > .015)
        error = abs(actual-pred)
        record = {"stroke": stroke.to_dict(), "overlap": overlap,
                  "execution": execution,
                  "prediction_union_gray_l1": float(error[union].mean()) if union.any() else 0.,
                  "prediction_full_gray_l1": float(error.mean()),
                  "union_pixels": int(union.sum())}
        return record, (before, pred, actual)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/force_model_stability")
    args = parser.parse_args()
    stage3, stage2, _, model, device, checkpoint = load_system()
    dynamics = LearnedCanvasDynamics(model, device, stage2, stage3)
    raw_config=copy.deepcopy(stage3)
    raw_config['convex_pigment_projection']=False
    raw_dynamics=LearnedCanvasDynamics(model,device,stage2,raw_config)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    configs = {name: json.loads((ROOT / path).read_text()) for name, path in (
        ("original", "configs/stage1_refined.json"), ("stable", "configs/stage1_stable.json"))}
    if stage3.get('convex_pigment_projection',False):
        configs['stable_raw']=copy.deepcopy(configs['stable'])
    forces = [.4, .45, .5, .7, 1., 1.3, 1.5]
    records = []
    fig, axes = plt.subplots(len(forces), 3, figsize=(10, 2.3*len(forces)))
    for path in range(2):
        for i, force in enumerate(forces):
            stroke = (Stroke(-.06, -.03, .055, .012, .1, force, i % 24) if path == 0
                      else Stroke(.015, -.04, .070, -.020, 2.1, force, (i*3+4) % 24))
            for name, stage1 in configs.items():
                adapter=raw_dynamics if name=='stable_raw' else dynamics
                rec, imgs = evaluate_case(stage1, stage3, adapter, stroke,
                                                seed=2026101200+path*10+i, overlap=path==1)
                rec.update(physics=name, path=path,
                           convex_pigment_projection=adapter.cfg.get('convex_pigment_projection',False))
                records.append(rec)
                if path == 0 and name == "stable":
                    yy, xx = np.where(abs(imgs[2]-imgs[0])>.015)
                    x0, x1 = max(0, xx.min()-12), min(512, xx.max()+13)
                    y0, y1 = max(0, yy.min()-12), min(512, yy.max()+13)
                    for ax, image, title in zip(axes[i], imgs, ("Before", "Learned prediction", "MuJoCo execution")):
                        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
                        ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis("off")
                        ax.set_title(f"{force:.2f} N | {title}",fontsize=10)
                print(f"{name} path={path} F={force:.2f}: force RMSE="
                      f"{rec['execution']['force_all_samples_rmse_N']:.4f}, "
                      f"union model L1={rec['prediction_union_gray_l1']:.4f}", flush=True)
    summary = {"protocol": "Two independent geometries, seven forces, blank and crossing-stroke canvases; no artwork targets",
               "model_configuration": stage3, "physics_configurations": configs, "samples": records,
               "summary": {name: {
                   "mean_force_all_samples_rmse_N": float(np.mean([r['execution']['force_all_samples_rmse_N'] for r in records if r['physics']==name])),
                   "worst_force_all_samples_rmse_N": float(max(r['execution']['force_all_samples_rmse_N'] for r in records if r['physics']==name)),
                   "minimum_contact_fraction": float(min(r['execution']['contact_fraction'] for r in records if r['physics']==name)),
                   "mean_prediction_union_gray_l1": float(np.mean([r['prediction_union_gray_l1'] for r in records if r['physics']==name]))}
                   for name in configs}}
    (out/"metrics.json").write_text(json.dumps(summary,indent=2))
    fig.tight_layout();fig.savefig(out/"thin_stroke_predictions.png",dpi=150);plt.close(fig)
    print(json.dumps(summary['summary'],indent=2),flush=True)


if __name__ == "__main__":
    main()
