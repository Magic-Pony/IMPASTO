"""Behavioural regression for low-force contact, independent of artworks."""
import json
from pathlib import Path

import pytest

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv


@pytest.mark.parametrize("force", [.4, .7, 1.3])
def test_stable_contact_retains_force_during_curved_strokes(force):
    config = json.loads(Path("configs/stage1_stable.json").read_text())
    env = PaintingEnv(config, seed=2026101200)
    try:
        result = env.execute_stroke(Stroke(-.06, -.03, .055, .012, .1, force, 0))
        assert result["success"]
        assert result["contact_fraction"] == 1.
        assert result["force_all_samples_rmse_N"] < .06
        assert result["trajectory_rmse_m"] < .0005
    finally:
        env.close()
