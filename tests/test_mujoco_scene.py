import json
from pathlib import Path

import mujoco
import numpy as np

from impasto.sim.mujoco_env import PaintingEnv


def config():
    return json.loads(Path("configs/stage1.json").read_text())


def test_scene_loads_with_panda_and_stage1_objects():
    env = PaintingEnv(config(), seed=3)
    try:
        assert env.model.nq == 9
        assert env.model.nu == 8
        assert env.model.ncam == 3
        assert env.model.nsite >= 4
        mujoco.mj_forward(env.model, env.data)
        assert np.isfinite(env.tip_position()).all()
    finally:
        env.close()


def test_two_environments_do_not_share_canvas_or_brush_state():
    a, b = PaintingEnv(config(), seed=1), PaintingEnv(config(), seed=2)
    try:
        a.brush.dip(2)
        a.canvas.update([0, 0], .8, a.brush, .01, contacting=True)
        assert b.brush.color == -1
        assert not b.canvas.changed_mask().any()
    finally:
        a.close(); b.close()


def test_brush_stays_vertical_during_workstation_moves():
    env = PaintingEnv(config(), seed=5)
    try:
        for target in ([.04, .235, .15], [.04, -.235, .099], [.45, -.03, .075]):
            for _ in range(250):
                env.step(target, phase="posture_test", paint=False)
            assert env.brush_tilt_deg() < 1.0
    finally:
        env.close()
