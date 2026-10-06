import json
from pathlib import Path

import mujoco
import numpy as np

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv


def config():
    return json.loads(Path("configs/stage1.json").read_text())


def test_scene_loads_with_panda_and_stage1_objects():
    env = PaintingEnv(config(), seed=3)
    try:
        assert env.model.nq == 10
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
        for target in ([.43, -.37, .18], [.43, .37, .18], [.50, .00, .075]):
            for _ in range(250):
                env.step(target, phase="posture_test", paint=False)
            assert env.brush_tilt_deg() < 1.0
    finally:
        env.close()


def test_all_workstations_have_vertical_ik_solutions():
    env = PaintingEnv(config())
    try:
        targets = [env.palette_target(i) for i in range(24)] + [env.washer_target()]
        for target in targets:
            q = env._ik(target, seed=env.home_q, max_iter=500)
            env.data.qpos[:7] = q
            mujoco.mj_forward(env.model, env.data)
            assert np.linalg.norm(env.data.site_xpos[env.control_site] - target) < .002
            assert env.brush_axis()[2] < -np.cos(np.deg2rad(1))
    finally:
        env.close()


def test_corner_wells_and_cleaner_are_physically_visited():
    env = PaintingEnv(config())
    try:
        for color in (0, 7, 16, 23):
            result = env.dip_brush(color)
            assert result['success'] and result['xy_error_m'] < .002
            assert env.brush.color == color
        assert env.clean_brush()['success']
        assert env.brush.color == -1
    finally:
        env.close()


def test_short_brush_stroke_keeps_contact_with_unscaled_newtons():
    env = PaintingEnv(config())
    try:
        result = env.execute_stroke(Stroke(-.06, -.04, .07, .012, .1, .6, 0))
        assert result['success']
        assert result['contact_fraction'] > .99
        assert result['force_all_samples_rmse_N'] < .08
        assert result['force_std_N'] < .05
        assert result['trajectory_rmse_m'] < .003
        assert env.normal_force() == env.normal_force_raw()
        assert env.tool_wrench().shape == (6,)
        assert np.isfinite(env.tool_wrench()).all()
    finally:
        env.close()
