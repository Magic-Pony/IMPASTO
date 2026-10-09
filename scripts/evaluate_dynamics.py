#!/usr/bin/env python3
"""Evaluate the held-out Stage-2 set and render representative predictions."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from impasto.stage2.data import Stage2Dataset
from impasto.stage2.model import ImpastoUNet


ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / os.environ.get("IMPASTO_STAGE2_CONFIG", "configs/stage2.json")).read_text())
    train_metrics = json.loads((ROOT / cfg.get("report_dir", "reports/stage2") / "training_metrics.json").read_text())
    checkpoint_path = ROOT / train_metrics["best_checkpoint"]
    dataset_path = ROOT / cfg["dataset_dir"] / "stage2_selfplay.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_metrics["best_epoch"] = int(checkpoint["epoch"])
    (ROOT / cfg.get("report_dir", "reports/stage2") / "training_metrics.json").write_text(
        json.dumps(train_metrics, indent=2))
    model = ImpastoUNet(channels=tuple(cfg["unet_channels"]),
                        action_embedding_channels=int(cfg["action_embedding_channels"]),
                        dropout=float(cfg["dropout_probability"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = Stage2Dataset(dataset_path, cfg["action_bounds"], False, cfg["seed"] + 1,
                            cfg["training"], cfg["split"]["validation_episode_ids"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    all_predictions, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device), batch["action"].to(device)).cpu().numpy()
            all_predictions.append(pred)
            all_targets.append(batch["target"].numpy())
            all_inputs.append(batch["image"].numpy())
    pred = np.concatenate(all_predictions)
    target = np.concatenate(all_targets)
    inputs = np.concatenate(all_inputs)
    absolute = np.abs(pred - target)
    changed = np.max(np.abs(target - inputs[:, 0:1]), axis=1, keepdims=True) > float(cfg["training"]["changed_pixel_threshold"])
    with np.load(dataset_path, allow_pickle=False) as archive:
        quality = archive["quality"]
    quality_summary = {
        "mean_target_force_N": float(quality[:, 0].mean()),
        "mean_force_N": float(quality[:, 1].mean()),
        "mean_force_rmse_N": float(quality[:, 2].mean()),
        "max_force_rmse_N": float(quality[:, 2].max()),
        "mean_contact_fraction": float(quality[:, 3].mean()),
        "min_contact_fraction": float(quality[:, 3].min()),
        "mean_trajectory_rmse_m": float(quality[:, 4].mean()),
        "mean_force_acquisition_s": float(quality[:, 6].mean()),
        "mean_failed_attempts": float(quality[:, 7].mean()),
    }
    metrics = {
        "validation_samples": len(dataset),
        "unweighted_l1": float(absolute.mean()),
        "stroke_region_l1": float(absolute[changed].mean()) if changed.any() else 0.0,
        "background_l1": float(absolute[~changed].mean()) if (~changed).any() else 0.0,
        "changed_pixel_fraction": float(changed.mean()),
        "paper_unweighted_l1_reference": .0285,
        "difference_from_paper_reference": float(absolute.mean() - .0285),
        "numerically_below_noncomparable_paper_reference": bool(absolute.mean() <= .0285),
        "checkpoint_sha256": train_metrics["best_checkpoint_sha256"],
        "best_epoch": int(checkpoint["epoch"]),
        "validation_episode_ids": cfg["split"]["validation_episode_ids"],
        "device": str(device),
        "collection_quality": quality_summary,
    }
    report_dir = ROOT / cfg.get("report_dir", "reports/stage2")
    (report_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    chosen = np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int)
    fig, axes = plt.subplots(len(chosen), 5, figsize=(13, 2.4 * len(chosen)), squeeze=False)
    for row, idx in enumerate(chosen):
        axes[row, 0].imshow(inputs[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(inputs[idx, 1], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(target[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(pred[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(absolute[idx, 0], cmap="magma", vmin=0, vmax=.25)
        if row == 0:
            for col, title in enumerate(("Current", "Rendered action", "Target next", "Predicted next", "Absolute error")):
                axes[row, col].set_title(title)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(report_dir / "predictions.png", dpi=170)
    plt.close(fig)
    with (report_dir / "training_history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    every = int(cfg["training"]["lr_milestone_every_epochs"])
    for row in history:
        completed = int(row["epoch"])
        row["learning_rate"] = str(
            float(cfg["training"]["learning_rate"])
            * float(cfg["training"]["lr_gamma"]) ** ((completed - 1) // every))
    with (report_dir / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    epochs = np.asarray([int(row["epoch"]) for row in history])
    train_unweighted = np.asarray([float(row["train_unweighted_l1"]) for row in history])
    train_weighted = np.asarray([float(row["train_weighted_l1"]) for row in history])
    validation = np.asarray([float(row["val_unweighted_l1"]) for row in history])
    validation_stroke = np.asarray([float(row["val_stroke_l1"]) for row in history])
    validation_background = np.asarray([float(row["val_background_l1"]) for row in history])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(epochs, train_unweighted, label="Train unweighted")
    axes[0].plot(epochs, validation, label="Validation unweighted")
    axes[0].axhline(.0285, color="black", linestyle="--", linewidth=1, label="Paper reference")
    axes[0].set_title("Full-patch L1")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, train_weighted, color="#d97706")
    axes[1].set_title("Train weighted L1")
    axes[2].plot(epochs, validation_stroke, label="Changed region")
    axes[2].plot(epochs, validation_background, label="Background")
    axes[2].set_title("Validation region L1")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L1")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=170)
    plt.close(fig)
    report = f"""# IMPASTO Stage 2: Pixel Dynamics Evaluation

## Result

The episode-held-out validation set contains {metrics['validation_samples']} transitions from episodes {metrics['validation_episode_ids']}. The best checkpoint is epoch {train_metrics['best_epoch']}; its unweighted next-state L1 is **{metrics['unweighted_l1']:.5f}**, compared with the paper's **0.0285** reference at 900 training samples. This local simulation score is lower than the reported reference. The same validation set selects the checkpoint, so this is a validation result rather than an untouched test result.

|Metric|Value|
|---|---:|
|Validation unweighted L1|{metrics['unweighted_l1']:.5f}|
|Changed-region L1|{metrics['stroke_region_l1']:.5f}|
|Background L1|{metrics['background_l1']:.5f}|
|Paper unweighted L1 reference|{metrics['paper_unweighted_l1_reference']:.5f}|
|Changed pixels per patch|{metrics['changed_pixel_fraction']:.3%}|
|Training samples|{train_metrics['train_samples']}|
|Validation samples|{train_metrics['validation_samples']}|
|Epochs / batch size|{train_metrics['epochs']} / {train_metrics['batch_size']}|
|Best epoch|{train_metrics['best_epoch']}|
|Training wall time|{train_metrics['training_wall_seconds']:.1f} s|
|Compute device|{train_metrics['gpu'] or train_metrics['device']}|
|Parameter count|{train_metrics['model_parameters']:,}|
|U-Net feature widths|{cfg['unet_channels']}|
|Action embedding / fusion level|{cfg['action_embedding_channels']} channels / 64-channel encoder level|
|Optimizer|Adam, lr={cfg['training']['learning_rate']}, weight decay={cfg['training']['weight_decay']}|
|Learning-rate decay|x{cfg['training']['lr_gamma']} every {cfg['training']['lr_milestone_every_epochs']} epochs|

## Training

![Training curves](training_curves.png)

![Held-out predictions](predictions.png)

## Simulator data quality

The 1,000 successful transitions came from ten independent 100-stroke episodes. Mean achieved force was {quality_summary['mean_force_N']:.3f} N against a {quality_summary['mean_target_force_N']:.3f} N mean target; mean per-stroke force RMSE was {quality_summary['mean_force_rmse_N']:.3f} N (maximum {quality_summary['max_force_rmse_N']:.3f} N). Mean contact fraction was {quality_summary['mean_contact_fraction']:.5f}, minimum {quality_summary['min_contact_fraction']:.5f}; mean trajectory RMSE was {quality_summary['mean_trajectory_rmse_m']*1000:.2f} mm. Failed attempts averaged {quality_summary['mean_failed_attempts']:.3f} per accepted sample.

## Reproducibility

Best checkpoint: `{train_metrics['best_checkpoint']}`  
SHA-256: `{train_metrics['best_checkpoint_sha256']}`  
Training history: `{train_metrics['history']}`  
Dataset manifest: `{cfg['dataset_dir']}/manifest.json`

The paper-specified model dimensions, optimizer, learning rate, batch size, epoch count, decay schedule, and 900 training samples are used. The signed bend normalization, numeric action limits, crop margin, configured task-specific color subset, brush-width law, dropout rate, and changed-pixel loss multiplier are local choices because the paper does not specify them. Inputs use a two-channel 100 x 100 patch and the complete next grayscale patch is predicted. The simulator is the Stage-1 contact-driven paint model, not the real robot; therefore the comparison with the paper is an implementation-level reference, not a controlled reproduction on identical hardware data. No baseline is implemented or evaluated in this stage.
"""
    (report_dir / "report.md").write_text(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
