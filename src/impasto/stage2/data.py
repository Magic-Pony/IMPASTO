"""Dataset, augmentation, and weighted-L1 helpers for Stage 2."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def weighted_l1_weights(current, target, threshold=.015, dilation_kernel=7, changed_weight=4.0):
    """Build normalized weights from a dilated frame difference mask."""
    diff = torch.amax(torch.abs(target - current), dim=1, keepdim=True)
    mask = (diff > threshold).float()
    pad = dilation_kernel // 2
    dilated = torch.nn.functional.max_pool2d(mask, dilation_kernel, stride=1, padding=pad)
    weights = 1.0 + changed_weight * dilated
    return weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def unweighted_l1(pred, target):
    return torch.mean(torch.abs(pred - target))


def transform_patch_action(action, patch_size, angle_deg=0.0, flip_x=False, flip_y=False,
                           translate=(0.0, 0.0)):
    """Apply the image augmentation to the quadratic Bezier action exactly."""
    action = np.asarray(action, dtype=np.float32)
    q0 = action[:2].copy()
    alpha = float(action[5])
    tangent = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    q1 = q0 + .5 * action[2] * tangent + action[3] * normal
    q2 = q0 + action[2] * tangent
    center = np.array([(patch_size - 1) / 2, (patch_size - 1) / 2], dtype=np.float32)
    th = np.deg2rad(angle_deg)
    # PIL's positive rotation turns a rightward image vector upward.
    rotation = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]], dtype=np.float32)
    sign = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0]).astype(np.float32)
    shift = np.asarray(translate, dtype=np.float32)
    def tf(p):
        return center + sign @ (rotation @ (p - center)) + shift
    nq0, nq1, nq2 = tf(q0), tf(q1), tf(q2)
    tangent = nq2 - nq0
    length = float(np.linalg.norm(tangent))
    if length < 1e-8:
        return action
    t = tangent / length
    n = np.array([-t[1], t[0]], dtype=np.float32)
    bend = float(np.dot(nq1 - nq0 - .5 * length * t, n))
    out = action.copy()
    out[0:2] = np.array([nq0[0], nq0[1]], dtype=np.float32)
    out[2] = length
    out[3] = bend
    out[4] = action[4]
    out[5] = np.arctan2(t[1], t[0])
    return out


class Stage2Dataset(Dataset):
    def __init__(self, path, bounds=None, augment=False, seed=0, config=None,
                 episode_ids=None):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        select = np.ones(len(data["images"]), dtype=bool)
        if episode_ids is not None:
            select = np.isin(data["episode_ids"], np.asarray(episode_ids))
        self.images = data["images"][select].astype(np.float32) / 255.0
        self.targets = data["targets"][select].astype(np.float32) / 255.0
        self.actions = data["actions"][select].astype(np.float32)
        self.raw_actions = data["raw_actions"][select].astype(np.float32)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.config = config or {}
        self.bounds = bounds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy()
        target = self.targets[index].copy()
        action = self.actions[index].copy()
        if self.augment:
            raw_action = self.raw_actions[index].copy()
            angle = float(self.rng.uniform(-self.config.get("augmentation_rotation_deg", 5.0),
                                           self.config.get("augmentation_rotation_deg", 5.0)))
            flip_x = bool(self.rng.integers(0, 2))
            flip_y = bool(self.rng.integers(0, 2))
            max_translation = float(self.config.get("augmentation_translation_px", 3.0))
            translate = tuple(self.rng.uniform(-max_translation, max_translation, size=2))
            center = ((image.shape[-1] - 1) / 2, (image.shape[-2] - 1) / 2)
            affine = cv2.getRotationMatrix2D(center, angle, 1.0)
            if flip_x:
                affine[0] *= -1
                affine[0, 2] += image.shape[-1] - 1
            if flip_y:
                affine[1] *= -1
                affine[1, 2] += image.shape[-2] - 1
            affine[:, 2] += translate
            image = np.stack([
                cv2.warpAffine(channel, affine, (channel.shape[1], channel.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(.955 if i == 0 else 1.0))
                for i, channel in enumerate(image)
            ])
            target = cv2.warpAffine(target[0], affine, (target.shape[-1], target.shape[-2]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=.955)[None]
            raw_action = transform_patch_action(raw_action, image.shape[-1], angle,
                                                flip_x, flip_y, translate)
            if self.bounds is not None:
                force_bounds = self.bounds["force_N"]
                scale = float(image.shape[-1] - 1)
                raw_action[[0, 1, 2, 3]] /= scale
                raw_action[3] = .5 + .5 * raw_action[3]
                raw_action[4] = (raw_action[4] - force_bounds[0]) / (force_bounds[1] - force_bounds[0])
                raw_action[5] = (raw_action[5] + np.pi) / (2 * np.pi)
                action = np.clip(raw_action, 0, 1)
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(target),
                "action": torch.from_numpy(action.astype(np.float32)), "index": index}


def save_manifest(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=True))
