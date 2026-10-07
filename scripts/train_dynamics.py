#!/usr/bin/env python3
"""Train IMPASTO-UNet with the paper's optimizer and weighted L1 objective."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from impasto.stage2.data import Stage2Dataset, weighted_l1_weights
from impasto.stage2.model import ImpastoUNet


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def evaluate(model, loader, device, cfg):
    model.eval()
    sums = {"l1": 0.0, "stroke_l1": 0.0, "background_l1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            pred = model(image, action)
            diff = torch.abs(pred - target)
            mask = torch.amax(torch.abs(target - image[:, 0:1]), dim=1, keepdim=True) > cfg["changed_pixel_threshold"]
            n = len(image)
            sums["l1"] += float(diff.mean(dim=(1, 2, 3)).sum())
            if mask.any():
                sums["stroke_l1"] += float((diff * mask).sum() / mask.sum().clamp_min(1)) * n
            background = ~mask
            if background.any():
                sums["background_l1"] += float((diff * background).sum() / background.sum().clamp_min(1)) * n
            count += n
    return {key: value / max(count, 1) for key, value in sums.items()}


def main():
    cfg = json.loads((ROOT / "configs/stage2.json").read_text())
    data_path = ROOT / cfg["dataset_dir"] / "stage2_selfplay.npz"
    report_dir = ROOT / "reports/stage2"
    output_dir = ROOT / "outputs/stage2"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Run collect_selfplay.py first: {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        required = ("images", "targets", "actions", "episode_ids", "step_ids", "quality")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {missing}")
        total = int(cfg["train_samples"]) + int(cfg["validation_samples"])
        if len(archive["images"]) != total or archive["images"].shape[1:] != (2, 100, 100):
            raise ValueError("Dataset image tensor has the wrong sample count or shape")
        if archive["targets"].shape != (total, 1, 100, 100):
            raise ValueError("Dataset target tensor has the wrong sample count or shape")
        if not np.isfinite(archive["actions"]).all() or not np.isfinite(archive["quality"]).all():
            raise ValueError("Dataset actions or measured quality values contain non-finite values")
        episode_ids = archive["episode_ids"]
        expected_ids = set(cfg["split"]["train_episode_ids"] + cfg["split"]["validation_episode_ids"])
        if set(map(int, np.unique(episode_ids))) != expected_ids:
            raise ValueError("Dataset episode IDs do not match the configured split")
        for episode_id in expected_ids:
            rows = np.flatnonzero(episode_ids == episode_id)
            if len(rows) != int(cfg["samples_per_episode"]):
                raise ValueError(f"Episode {episode_id} has an unexpected sample count")
            if not np.array_equal(archive["step_ids"][rows], np.arange(len(rows))):
                raise ValueError(f"Episode {episode_id} has invalid step ordering")
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids = cfg["split"]["train_episode_ids"]
    val_ids = cfg["split"]["validation_episode_ids"]
    train_set = Stage2Dataset(data_path, cfg["action_bounds"], True, cfg["seed"],
                              cfg["training"], train_ids)
    val_set = Stage2Dataset(data_path, cfg["action_bounds"], False, cfg["seed"] + 1,
                            cfg["training"], val_ids)
    if len(train_set) != int(cfg["train_samples"]) or len(val_set) != int(cfg["validation_samples"]):
        raise ValueError(f"Expected 900/100 train/validation samples, got {len(train_set)}/{len(val_set)}")
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_set, batch_size=int(cfg["training"]["batch_size"]),
                              shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=int(cfg["training"]["batch_size"]),
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ImpastoUNet(channels=tuple(cfg["unet_channels"]),
                        action_embedding_channels=int(cfg["action_embedding_channels"]),
                        dropout=float(cfg["dropout_probability"])).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                           weight_decay=float(cfg["training"]["weight_decay"]))
    every = int(cfg["training"]["lr_milestone_every_epochs"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(range(every, int(cfg["training"]["epochs"]), every)),
        gamma=float(cfg["training"]["lr_gamma"]))
    best_path = output_dir / "impasto_unet_best.pt"
    last_path = output_dir / "impasto_unet_last.pt"
    history_path = report_dir / "training_history.csv"
    history = []
    best_val = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        epoch_learning_rate = float(opt.param_groups[0]["lr"])
        weighted_sum = 0.0
        unweighted_sum = 0.0
        count = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            pred = model(image, action)
            weights = weighted_l1_weights(
                image[:, 0:1], target,
                threshold=float(cfg["training"]["changed_pixel_threshold"]),
                dilation_kernel=int(cfg["training"]["dilation_kernel"]),
                changed_weight=float(cfg["training"]["changed_pixel_weight"]))
            loss = (weights * torch.abs(pred - target)).sum(dim=(1, 2, 3)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            weighted_sum += float(loss.detach()) * len(image)
            unweighted_sum += float(torch.abs(pred.detach() - target).mean()) * len(image)
            count += len(image)
        scheduler.step()
        val = evaluate(model, val_loader, device, cfg["training"])
        row = {"epoch": epoch, "train_weighted_l1": weighted_sum / count,
               "train_unweighted_l1": unweighted_sum / count,
               "val_unweighted_l1": val["l1"], "val_stroke_l1": val["stroke_l1"],
               "val_background_l1": val["background_l1"],
               "learning_rate": epoch_learning_rate}
        history.append(row)
        if val["l1"] < best_val:
            best_val = val["l1"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation": val, "config": cfg,
                        "torch_version": torch.__version__}, best_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["training"]["epochs"]):
            with history_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            print(json.dumps({**row, "best_val_l1": best_val,
                              "epoch_wall_s": (time.perf_counter() - started) / epoch}), flush=True)
    torch.save({"state_dict": model.state_dict(), "epoch": int(cfg["training"]["epochs"]),
                "config": cfg, "torch_version": torch.__version__}, last_path)
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "model_parameters": params,
        "train_samples": len(train_set), "validation_samples": len(val_set),
        "epochs": int(cfg["training"]["epochs"]), "batch_size": int(cfg["training"]["batch_size"]),
        "best_epoch": best_epoch,
        "best_validation_unweighted_l1": best_val,
        "paper_validation_unweighted_l1": .0285,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best_path),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "training_wall_seconds": time.perf_counter() - started,
        "history": str(history_path.relative_to(ROOT)),
    }
    (report_dir / "training_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
