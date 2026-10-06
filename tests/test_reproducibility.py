import json
from pathlib import Path

import numpy as np

from impasto.actions import sample_strokes
from impasto.sim.mujoco_env import PaintingEnv


def test_seeded_stroke_sampler_is_reproducible():
    a = [s.to_dict() for s in sample_strokes(20, 4)]
    b = [s.to_dict() for s in sample_strokes(20, 4)]
    assert a == b


def test_environment_reset_has_finite_dynamics():
    cfg = json.loads(Path("configs/stage1.json").read_text())
    env = PaintingEnv(cfg, seed=10)
    try:
        for _ in range(20):
            env.step([.50, 0.0, .15], phase="test", paint=False)
        assert np.isfinite(env.data.qpos).all()
        assert np.isfinite(env.data.qvel).all()
        assert np.isfinite(env.tip_position()).all()
    finally:
        env.close()
