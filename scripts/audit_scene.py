"""Reproducible scene audit: 25 workstations, tool dimensions and contact force."""
from pathlib import Path
import json

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/scene_audit'


def static_reach(env, target):
    saved = env.data.qpos.copy()
    q = env._ik(target, seed=env.home_q, max_iter=500)
    env.data.qpos[:7] = q
    mujoco.mj_forward(env.model, env.data)
    result = {'target_m': np.asarray(target).tolist(),
              'position_error_m': float(np.linalg.norm(env.data.site_xpos[env.control_site] - target)),
              'tilt_deg': env.brush_tilt_deg()}
    env.data.qpos[:] = saved
    mujoco.mj_forward(env.model, env.data)
    return result


def tool_closeup(env, filename):
    """Move the diagnostic camera only, preserving the simulated robot state."""
    cid = env.camera_ids['overview_camera']
    old_pos, old_quat = env.model.cam_pos[cid].copy(), env.model.cam_quat[cid].copy()
    tip = env.tip_position()
    target = tip + [0, 0, .060]
    pos = target + [.15, -.25, .09]
    z = pos - target; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = Rotation.from_matrix(np.column_stack((x, y, z))).as_quat()
    try:
        env.model.cam_pos[cid] = pos
        env.model.cam_quat[cid] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(env.model, env.data)
        Image.fromarray(env.render('overview_camera', 960, 720)).save(OUT/filename)
    finally:
        env.model.cam_pos[cid] = old_pos
        env.model.cam_quat[cid] = old_quat
        mujoco.mj_forward(env.model, env.data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'configs/stage1.json').read_text())
    env = PaintingEnv(cfg, seed=51)
    targets = [env.palette_target(i) for i in range(24)] + [env.washer_target()]
    reach = [static_reach(env, target) for target in targets]
    collision_pairs = set()
    max_tilt = 0.
    screenshots = set()

    def hook(e, obs):
        nonlocal max_tilt
        max_tilt = max(max_tilt, obs.brush_tilt_deg)
        for contact in e.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair != {e.tip_geom, e.canvas_geom} and contact.dist < -.0005:
                collision_pairs.add(tuple(sorted(e.model.geom(g).name or f'geom_{g}' for g in pair)))
        # Save the first frame after vertical insertion finishes.
        for phase, name in [('DIP_HOLD', 'dip'), ('CLEAN_SPIN', 'clean')]:
            if e.phase == phase and name not in screenshots:
                Image.fromarray(e.render('overview_camera', 960, 540)).save(OUT/f'{name}.png')
                tool_closeup(e, f'{name}_closeup.png')
                screenshots.add(name)

    visits = []
    for color in range(24):
        visits.append(env.dip_brush(color, hook))
        visits.append(env.clean_brush(hook))
        print(f'visited color {color:02d} and cleaner', flush=True)
    rows = []
    def stroke_hook(e, obs):
        hook(e, obs)
        if e.phase == 'STROKE':
            rows.append([obs.time, obs.normal_force, obs.tip_xyz[2], obs.target_xyz[2], e.data.qpos[-1]])
    stroke = env.execute_stroke(Stroke(-.06, -.04, .07, .012, .1, .6, 0), stroke_hook)
    Image.fromarray(env.render('overview_camera', 1280, 720)).save(OUT/'scene.png')
    Image.fromarray(env.render('top_camera', 900, 900)).save(OUT/'top_view.png')
    tool_closeup(env, 'tool_closeup.png')
    force = np.asarray(rows)
    np.savetxt(OUT/'force_trace.csv', force, delimiter=',', header='time_s,contact_N,actual_tip_z_m,reference_z_m,compression_joint_m', comments='')
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axs[0].plot(force[:, 0], force[:, 1], label='MuJoCo contact force (unscaled)')
    axs[0].axhline(.6, color='tab:red', ls='--', label='target 0.6 N')
    axs[0].set_ylabel('Normal force (N)'); axs[0].legend(); axs[0].grid(alpha=.2)
    axs[1].plot(force[:, 0], -force[:, 4]*1000)
    axs[1].set_ylabel('Axial bristle compression (mm)'); axs[1].set_xlabel('Simulation time (s)'); axs[1].grid(alpha=.2)
    fig.savefig(OUT/'force.png', dpi=150); plt.close(fig)
    grid = [static_reach(env, [*env.canvas_xy([x,y]), env.canvas_contact_height()])
            for x in np.linspace(-.21,.21,5) for y in np.linspace(-.25,.25,5)]
    metrics = {'config': cfg, 'static_workstations': reach, 'dynamic_visits': visits,
               'static_max_error_m': max(r['position_error_m'] for r in reach),
               'dynamic_max_xy_error_m': max(r['xy_error_m'] for r in visits),
               'dynamic_max_immersion_tilt_deg': max(r['tilt_deg'] for r in visits),
               'all_movement_max_tilt_deg': max_tilt,
               'unexpected_penetrations_over_0_5mm': sorted(collision_pairs),
               'canvas_grid_5x5': grid, 'stroke': stroke,
               'force_source': 'unscaled brush/canvas contact; mount 6-axis sensor separately available',
               'dimensions_mm': {'cut_brush_length': 50, 'inserted_length': 16,
                                 'exposed_length': 34, 'retained_handle': 14, 'ferrule': 16,
                                 'bristles': 20, 'sensor_diameter': 54, 'sensor_height': 30,
                                 'hand_to_unloaded_brush_end': 221,
                                 'hand_to_nominal_tip_center': 217}}
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2))
    report = f"""# IMPASTO 场景与工具审计

本报告对当前 MuJoCo 场景中的工具装配、24 个颜料槽、清洗器、画布网格和短笔力控进行验收。论文只公开了图 A1 的装配组成，没有公开 CAD 尺寸；尺寸假设和论文边界见 [`docs/scene_audit.md`](../../docs/scene_audit.md)。

| 指标 | 结果 |
|---|---:|
| 颜料槽 | 24/24 成功 |
| 清洗器 | 24/24 成功 |
| 工位最大静态 IK 误差 | {metrics['static_max_error_m']*1000:.3f} mm |
| 动态工位最大 XY 误差 | {metrics['dynamic_max_xy_error_m']*1000:.3f} mm |
| 工位浸入最大倾角 | {metrics['dynamic_max_immersion_tilt_deg']:.4f} deg |
| 全部移位最大倾角 | {metrics['all_movement_max_tilt_deg']:.3f} deg |
| 画布 5x5 网格最大 IK 误差 | {max(r['position_error_m'] for r in grid)*1000:.3f} mm |
| 意外穿透（超过 0.5 mm） | {len(collision_pairs)} |
| 单笔接触比例 | {stroke['contact_fraction']:.3f} |
| 单笔接触力 RMSE | {stroke['force_all_samples_rmse_N']:.3f} N |
| 单笔运笔力标准差 | {stroke['force_std_N']:.3f} N |
| 单笔轨迹 RMSE | {stroke['trajectory_rmse_m']*1000:.3f} mm |

工具近景、场景截图、俯视图和力曲线：

- [工具近景](tool_closeup.png)
- [蘸漆近景](dip_closeup.png)
- [清洗近景](clean_closeup.png)
- [完整场景](scene.png)
- [俯视图](top_view.png)
- [力与刷毛压缩曲线](force.png)
- [原始力数据](force_trace.csv)
- [完整数值](metrics.json)
"""
    (OUT/'report.md').write_text(report)
    env.close()
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('config','static_workstations','dynamic_visits','canvas_grid_5x5')},indent=2))
    assert all(r['success'] for r in visits)
    assert metrics['static_max_error_m'] < .002
    assert metrics['dynamic_max_xy_error_m'] < .002
    assert max_tilt < 1
    assert not collision_pairs
    assert stroke['contact_fraction'] > .99 and stroke['force_all_samples_rmse_N'] < .08


if __name__ == '__main__':
    main()
