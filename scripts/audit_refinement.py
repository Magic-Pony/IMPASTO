#!/usr/bin/env python3
"""Generate inspectable, consistently cropped diagnostics from saved rollouts.

No planning, execution or target pixels are changed by this script. The masks
and metrics supplement, rather than replace, visual inspection of each round.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from impasto.actions import Stroke
from impasto.stage2.raster import srgb_gray
from impasto.stage3.targets import render_target_prefix

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = np.array([245, 242, 234], dtype=np.float32)


def foreground(rgb):
    return np.max(np.abs(np.asarray(rgb, dtype=np.float32) - BACKGROUND), axis=-1) > 15


def region_metrics(actual, target):
    actual, target = np.asarray(actual), np.asarray(target)
    a, t = foreground(actual), foreground(target)
    union = a | t
    error = np.mean(np.abs(actual.astype(float) - target.astype(float)), axis=-1) / 255
    gray_error = np.abs(srgb_gray(actual) - srgb_gray(target))
    return {
        "foreground_iou": float((a & t).sum() / max(1, union.sum())),
        "target_foreground_recall": float((a & t).sum() / max(1, t.sum())),
        "paint_precision": float((a & t).sum() / max(1, a.sum())),
        "target_foreground_rgb_l1": float(error[t].mean()),
        "union_foreground_rgb_l1": float(error[union].mean()),
        "target_foreground_gray_l1": float(gray_error[t].mean()),
        "foreground_threshold_rgb_255": 15,
    }


def bounds(rgb):
    yy, xx = np.nonzero(foreground(rgb))
    return (max(0, xx.min()-18), min(rgb.shape[1], xx.max()+19),
            max(0, yy.min()-18), min(rgb.shape[0], yy.max()+19))


def image_axis(ax, array, title, box, gray=False):
    if gray:
        ax.imshow(array, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(array)
    x0, x1, y0, y1 = box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def draw_actions(ax, actions, colors, palette, config, prefix, count=None):
    for i, (action, color) in enumerate(zip(actions, colors)):
        if count is not None and i >= count:
            break
        pts = Stroke(*map(float, action)).points(np.linspace(0, 1, 100))
        px = (pts[:, 0] / config["canvas_width_m"] + .5) * 511
        py = (.5 - pts[:, 1] / config["canvas_height_m"]) * 511
        ax.plot(px, py, color="white", lw=4)
        ax.plot(px, py, color=np.array(palette[int(color)])/255, lw=2)
        ax.plot(px[0], py[0], "o", ms=3, color="black")
        ax.text(px[0]+2, py[0]-2, f"{prefix}{i+1}", fontsize=8,
                color="black", bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))


def audit(report_dir, out_dir, previous_report=None, rgb_prediction=False):
    config = json.loads((report_dir / "resolved_config.json").read_text())
    metrics = json.loads((report_dir / "metrics.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"source_report": str(report_dir.relative_to(ROOT)), "experiments": {}}
    dynamics=rgb_planner=None
    if rgb_prediction:
        from scripts.run_stage3 import load_system
        from impasto.stage3.planner import LearnedCanvasDynamics,AdaptiveMPCPlanner
        cfg,s2,_,model,device,_=load_system(str(report_dir/'resolved_config.json'))
        actual_hash=hashlib.sha256((ROOT/cfg['checkpoint']).read_bytes()).hexdigest()
        if actual_hash!=metrics['paper_protocol']['checkpoint_sha256']:
            raise ValueError('RGB audit checkpoint differs from the recorded run')
        dynamics=LearnedCanvasDynamics(model,device,s2,cfg)
        rgb_planner=AdaptiveMPCPlanner(dynamics,cfg)
    for name in ("flower", "fish"):
        if name not in metrics["experiments"]:
            continue
        exp = metrics["experiments"][name]
        final_target = np.asarray(Image.open(report_dir / f"{name}_target.png"))
        final = np.asarray(Image.open(report_dir / f"{name}_after_{exp['executed_strokes']:02d}.png"))
        box = bounds(final_target)
        audited = {"final": region_metrics(final, final_target), "cycles": [], "crop": list(map(int, box))}
        fig, axes = plt.subplots(len(exp["cycles"]), 4, figsize=(11, 2.65*len(exp["cycles"])))
        for i, cycle in enumerate(exp["cycles"]):
            data_path = report_dir / f"{name}_cycle_{i+1:02d}.npz"
            with np.load(data_path) as d:
                d = {k: d[k] for k in d.files}
            before, target, after = d["before_rgb"], d["target_rgb"], d["after_rgb"]
            predicted, terminal = d["predicted"], d["predicted_terminal"]
            count = cycle["prediction_execution"]["executed_actions"]
            start = cycle["executed_before_replan"]
            goal = cycle["goal_stroke_count"]
            p_error = np.abs(predicted-srgb_gray(after))
            # Fixed target+paint union avoids averaging over mostly blank canvas.
            union = foreground(target) | foreground(before) | foreground(after)
            rec = {"round": i+1, "before": start, "after": start+count,
                   "goal": goal, "horizon": len(d["actions"]),
                   "prediction_execution_union_gray_l1": float(p_error[union].mean()),
                   "before_metrics": region_metrics(before, target),
                   "after_metrics": region_metrics(after, target),
                   "source_npz_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                   "actions": d["actions"][:count].tolist(),
                   "colors": d["colors"][:count].tolist()}
            audited["cycles"].append(rec)
            if rgb_prediction:
                initial=srgb_gray(before)[None]
                if config.get('rgb_equal_luminance_coverage',False):
                    _,frames,coverages=dynamics.rollout_batch(initial,d['actions'][None],
                        d['colors'][None],d['opacities'][None],return_coverage=True)
                else:
                    _,frames=dynamics.rollout_batch(initial,d['actions'][None],
                        d['colors'][None],d['opacities'][None])
                    coverages=[None]*len(frames)
                max_difference=float(abs(frames[count-1][0]-predicted).max())
                if max_difference>1e-4:
                    raise ValueError(f'Recomputed grayscale does not match archive: {max_difference}')
                rgb=before[None].astype(np.float32)/255
                previous_gray=initial
                rgb_frames=[]
                for t,frame in enumerate(frames):
                    rgb=rgb_planner._rgb_step(rgb,previous_gray,frame,[d['colors'][t]],coverages[t])
                    previous_gray=frame
                    rgb_frames.append(rgb[0].copy())
                    if t==count-1:
                        rgb_after=rgb[0].copy()
                error_rgb=np.mean(abs(rgb_after-after.astype(float)/255),axis=-1)
                rec['prediction_execution_union_rgb_l1']=float(error_rgb[union].mean())
                rec['rgb_recomputed_gray_max_difference']=max_difference
                np.savez_compressed(out_dir/f'{name}_rgb_prediction_{i+1:02d}.npz',
                    after_actions=rgb_after,terminal=rgb[0],frames=np.asarray(rgb_frames))
                color_fig,color_axes=plt.subplots(1,4,figsize=(14,4))
                for axis,arr,title in zip(color_axes,[target,rgb[0],rgb_after,after],
                    ['Checkpoint target','Predicted RGB terminal','Predicted RGB executed prefix','MuJoCo executed prefix']):
                    image_axis(axis,arr,title,box)
                color_fig.tight_layout()
                color_fig.savefig(out_dir/f'{name}_rgb_audit_{i+1:02d}.png',dpi=145)
                plt.close(color_fig)
                sequence_fig,sequence_axes=plt.subplots(1,len(rgb_frames)+2,
                    figsize=(3*(len(rgb_frames)+2),3.5))
                image_axis(sequence_axes[0],before,'Observed before',box)
                for t,frame in enumerate(rgb_frames,1):
                    image_axis(sequence_axes[t],frame,
                        f'Predict action {t}'+(' (executed)' if t<=count else ' (not executed)'),box)
                image_axis(sequence_axes[-1],after,f'Actual after {count}',box)
                sequence_fig.tight_layout()
                sequence_fig.savefig(out_dir/f'{name}_rgb_rollout_{i+1:02d}.png',dpi=145)
                plt.close(sequence_fig)
            fig2, ax = plt.subplots(2, 4, figsize=(14, 7.5))
            ax = ax.ravel()
            image_axis(ax[0], target, f"Target at stroke {goal}", box)
            image_axis(ax[1], before, f"Observed after {start} strokes", box)
            seed_image = target.copy().astype(float)/255
            for j, mask in enumerate(d["seed_masks"]):
                seed_image[mask] = .35*seed_image[mask] + .65*np.array(plt.get_cmap("tab10")(j%10)[:3])
            image_axis(ax[2], seed_image, "PCA/RANSAC masks + seed paths", box)
            draw_actions(ax[2], d["seeds"], d["seed_colors"], config["palette_rgb"], config, "S")
            image_axis(ax[3], target, f"Optimized actions (execute first {count})", box)
            draw_actions(ax[3], d["actions"], d["colors"], config["palette_rgb"], config, "A", count)
            image_axis(ax[4], terminal, f"Predicted terminal: H={len(d['actions'])}", box, True)
            image_axis(ax[5], predicted, f"Predicted after {count} actions", box, True)
            image_axis(ax[6], after, f"Executed after same {count} actions", box)
            ax[7].imshow(p_error, cmap="magma", vmin=0, vmax=.15)
            ax[7].set_xlim(box[:2]); ax[7].set_ylim(box[3], box[2]); ax[7].axis("off")
            ax[7].set_title(f"Prediction error (0-0.15)\nunion L1={p_error[union].mean():.4f}", fontsize=11)
            fig2.suptitle(f"{name.title()} round {i+1}: {start} -> {start+count} / checkpoint {goal}", fontsize=14)
            fig2.tight_layout()
            fig2.savefig(out_dir / f"{name}_audit_{i+1:02d}.png", dpi=145)
            plt.close(fig2)
            for axis, arr, title, gray in zip(axes[i], [target, before, predicted, after],
                    [f"R{i+1} target {goal}", f"Before {start}", f"Predict {start+count}", f"Execute {start+count}"],
                    [False, False, True, False]):
                image_axis(axis, arr, title, box, gray)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_rounds_montage.png", dpi=120)
        plt.close(fig)
        results["experiments"][name] = audited
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        image_axis(axes[0], final_target, "Published target (normalized crop)", box)
        previous_dir = previous_report or ROOT / "reports/stage3_refined_attempt6"
        previous = previous_dir / f"{name}_after_{exp['executed_strokes']:02d}.png"
        if previous.exists():
            previous_image = np.asarray(Image.open(previous))
            image_axis(axes[1], previous_image, f"Previous: {previous_dir.name}", box)
            audited["previous_on_current_target"] = region_metrics(previous_image, final_target)
            audited["previous_report"] = str(previous_dir.relative_to(ROOT))
        image_axis(axes[2], final, "Refined execution", box)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}_comparison.png", dpi=170); plt.close(fig)
    (out_dir / "visual_audit_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/stage3_refined")
    parser.add_argument("--out-dir", default="reports/refinement")
    parser.add_argument("--previous-report", help="Local run to compare on the current target")
    parser.add_argument('--rgb-prediction',action='store_true',help='Recompute RGB rollout using the recorded checkpoint')
    args = parser.parse_args()
    result = audit(ROOT/args.report_dir, ROOT/args.out_dir,
                   ROOT/args.previous_report if args.previous_report else None,
                   rgb_prediction=args.rgb_prediction)
    print(json.dumps({k: v["final"] for k, v in result["experiments"].items()}, indent=2))
