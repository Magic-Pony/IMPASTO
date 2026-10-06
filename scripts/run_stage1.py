#!/usr/bin/env python3
"""Run the Stage-1 environment validation and build its report artifacts."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv, write_episode
from impasto.sim.paint import PALETTE


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stage1"
FIG = REPORT / "figures"
VID = REPORT / "videos"
DATA = REPORT / "data"
CONFIG_PATH = ROOT / "configs/stage1.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def overview_frame(env, camera="overview_camera", size=(640, 360), title=""):
    rgb = env.render(camera, width=size[0], height=size[1])
    base = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(base)
    if title:
        draw.rectangle((0, 0, min(size[0], 520), 31), fill=(0, 0, 0))
        draw.text((8, 6), title, fill=(255, 255, 255), font=font(17))
    return np.asarray(base)


def topdown_frame(env, size=(640, 480), title=""):
    # This is the simulated camera image, including the live canvas texture.
    im = Image.fromarray(env.render("canvas_camera", width=size[0], height=size[1])).convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, size[0]-1, size[1]-1), outline=(25, 25, 25), width=4)
    if title:
        draw.rectangle((0, 0, min(size[0], 450), 31), fill=(0, 0, 0))
        draw.text((8, 6), title, fill=(255, 255, 255), font=font(17))
    return np.asarray(im)


def flower_strokes():
    return [
        Stroke(-.025, -.13, .15, .012, np.pi/2, .55, color=12),
        Stroke(.05, .01, .075, .012, np.pi/3, .60, color=1),
        Stroke(.05, .01, .075, -.012, -np.pi/3, .60, color=1),
        Stroke(.05, .01, .078, .010, 0.10, .65, color=1),
        Stroke(.05, .01, .070, -.012, 2.15, .60, color=1),
        Stroke(.05, .01, .070, .010, -2.15, .60, color=1),
    ]


def fish_strokes():
    return [
        Stroke(-.105, .045, .16, .020, 0.05, .55, color=0),
        Stroke(-.03, .075, .065, -.010, 2.30, .50, color=1),
        Stroke(-.03, .015, .065, .010, -2.30, .50, color=1),
        Stroke(.045, .045, .045, .010, np.pi/2, .70, color=5),
    ]


def run_demo(config, seed, strokes, video_name=None):
    env = PaintingEnv(config, seed=seed, render=video_name is not None)
    render_fps = float(config.get("render_fps", 24))
    writer_overview = writer_top = None
    if video_name:
        opts = dict(fps=render_fps, codec="libx264", macro_block_size=None)
        writer_overview = imageio.get_writer(VID / f"{video_name}_scene.mp4", **opts)
        writer_top = imageio.get_writer(VID / f"{video_name}_topdown.mp4", **opts)
    frame_period = 1.0 / max(render_fps, 1.0)
    next_frame_time = 0.0

    def hook(e, obs):
        nonlocal next_frame_time
        if video_name and obs.time + 1e-9 >= next_frame_time:
            title = f"Stage 1 | {e.phase} | t={obs.time:.2f}s | F={obs.normal_force:.2f} N"
            writer_overview.append_data(overview_frame(e, title=title))
            writer_top.append_data(topdown_frame(e, title=f"canvas | {e.phase}"))
            # Match the sampling comparison's tolerance when advancing the
            # clock, avoiding duplicate frames at floating-point boundaries.
            next_frame_time = (np.floor((obs.time + 1e-9) / frame_period) + 1.0) * frame_period

    wall_start = time.perf_counter()
    try:
        results = [env.execute_stroke(s, render_hook=hook) for s in strokes]
        if video_name:
            final_overview = overview_frame(env, title=f"Stage 1 complete | t={env.data.time:.2f}s")
            final_top = topdown_frame(env, title="canvas | complete")
            for _ in range(10):
                writer_overview.append_data(final_overview)
                writer_top.append_data(final_top)
    finally:
        if writer_overview is not None:
            writer_overview.close()
            writer_top.close()
    wall = time.perf_counter() - wall_start
    return env, results, wall


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def stage_plots(config, env, results, multi_strokes):
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # Scene layout combines the actual MuJoCo camera and the explicit paint state.
    scene = Image.new("RGB", (1000, 520), (230, 230, 230))
    scene.paste(Image.fromarray(env.render("overview_camera", 640, 360)), (0, 80))
    scene.paste(Image.fromarray(env.render("canvas_camera", 340, 340)), (650, 80))
    d = ImageDraw.Draw(scene)
    d.text((15, 20), "MuJoCo overview", fill=(20, 20, 20), font=font(24))
    d.text((665, 20), "Canvas camera", fill=(20, 20, 20), font=font(24))
    scene.save(FIG / "scene_layout.png")

    before = Image.new("RGB", (512, 256), "white")
    before.paste(Image.fromarray(env.canvas.initial.astype(np.uint8) if env.canvas.initial.max() > 1 else np.uint8(env.canvas.initial*255)).resize((256, 256)), (0, 0))
    before.paste(Image.fromarray(env.canvas.image()).resize((256, 256)), (256, 0))
    bd = ImageDraw.Draw(before)
    bd.text((8, 8), "before", fill=(0, 0, 0), font=font(18))
    bd.text((264, 8), "after", fill=(0, 0, 0), font=font(18))
    before.save(FIG / "canvas_before_after.png")

    # Derive phase labels from transition timestamps.
    phase_ranges = []
    stroke_index = -1
    for i, ev in enumerate(env.events):
        if ev["phase"] == "MOVE_TO_START":
            stroke_index += 1
        t0 = ev["time"]
        t1 = env.events[i+1]["time"] if i + 1 < len(env.events) else float("inf")
        force_target = (multi_strokes[stroke_index].force
                        if 0 <= stroke_index < len(multi_strokes) else "")
        phase_ranges.append((t0, t1, ev["phase"], force_target))
    def phase_at(t):
        for t0, t1, p, target_force in phase_ranges:
            if t0 <= t < t1:
                return p, target_force
        return "unknown", ""

    observations = [(o, *phase_at(o.time)) for o in env.observations]
    write_csv(DATA / "force_trace.csv", ["time_s", "phase", "force_N", "contacting", "target_force_N"],
              [[o.time, p, o.normal_force, int(o.contacting), target]
               for o, p, target in observations])
    write_csv(DATA / "trajectory.csv", ["time_s", "phase", "tip_x_m", "tip_y_m", "tip_z_m", "target_x_m", "target_y_m", "target_z_m"],
              [[o.time, p, *o.tip_xyz, *o.target_xyz] for o, p, _ in observations])

    plt.figure(figsize=(10, 4))
    for o, p, _ in observations:
        if p in ("FORCE_ACQUIRE", "STROKE"):
            plt.scatter(o.time, o.normal_force, s=2, c="tab:blue" if p == "STROKE" else "tab:orange")
    stroke_targets = [r.get("target_force_N", np.nan) for r in results if r.get("success")]
    if stroke_targets:
        plt.axhline(float(np.mean(stroke_targets)), color="tab:red", ls="--", label=f"mean target {np.mean(stroke_targets):.2f} N")
    plt.xlabel("simulation time (s)"); plt.ylabel("brush/canvas contact force (N)"); plt.title("Force acquisition and stroke tracking")
    plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(FIG / "force_tracking.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 7))
    for s in multi_strokes:
        pts = s.points(np.linspace(0, 1, 100))
        plt.plot(pts[:, 0], pts[:, 1], "--", lw=1.5, label=f"target c{s.color}")
    for o, p, _ in observations:
        if p == "STROKE" and o.contacting:
            plt.scatter(o.tip_xyz[0]-.5, o.tip_xyz[1], c="k", s=2)
    plt.gca().set_aspect("equal"); plt.xlim(-.23, .23); plt.ylim(-.27, .27)
    plt.xlabel("canvas x (m)"); plt.ylabel("canvas y (m)"); plt.title("Target Bezier strokes and contacted tip samples")
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG / "trajectory.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 3))
    cols = PALETTE
    plt.imshow(cols.reshape(3, 8, 3), aspect="auto")
    for i in range(24):
        r, c = divmod(i, 8); plt.text(c, r, str(i), ha="center", va="center", color="white" if cols[i].mean()<.52 else "black", fontsize=8)
    plt.yticks([]); plt.xticks([]); plt.title("24 simulated premixed pigment wells; cleaner is an explicit state transition")
    plt.tight_layout(); plt.savefig(FIG / "palette_and_cleaning.png", dpi=160); plt.close()


def worker(task):
    config, seed = task
    env = PaintingEnv(config, seed=seed, render=False)
    # Use the same reachable, centered workload for the serial/parallel
    # benchmark so it measures process throughput rather than workspace-edge
    # IK failures. Random self-play sampling is exercised separately in the
    # action/data tests and will be expanded in Stage 2.
    stroke = Stroke(-.05, -.03, .06, 0.0, 0.0, .60, color=seed % 24)
    t0 = time.perf_counter()
    result = env.execute_stroke(stroke)
    wall = time.perf_counter() - t0
    return {"seed": seed, "success": result["success"], "wall_s": wall,
            "sim_s": float(env.data.time), "pixels": int(env.canvas.changed_mask().sum())}


def throughput(config, workers=4, count=8):
    tasks = [(config, 7000+i) for i in range(count)]
    t0 = time.perf_counter(); serial = [worker(t) for t in tasks]; serial_wall = time.perf_counter()-t0
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        parallel = list(pool.map(worker, tasks))
    parallel_wall = time.perf_counter()-t0
    out = {"count": count, "workers": workers, "serial_wall_s": serial_wall,
           "parallel_wall_s": parallel_wall, "serial_strokes_per_s": count/serial_wall,
           "parallel_strokes_per_s": count/parallel_wall, "speedup": serial_wall/parallel_wall,
           "serial_success_rate": float(np.mean([x["success"] for x in serial])),
           "parallel_success_rate": float(np.mean([x["success"] for x in parallel]))}
    rows = [["serial", serial_wall, count/serial_wall, out["serial_success_rate"]],
            ["parallel", parallel_wall, count/parallel_wall, out["parallel_success_rate"]]]
    write_csv(DATA / "throughput.csv", ["mode", "wall_s", "strokes_per_s", "success_rate"], rows)
    plt.figure(figsize=(6, 4)); plt.bar(["serial", "parallel"], [count/serial_wall, count/parallel_wall], color=["#999999", "#2277aa"])
    plt.ylabel("completed strokes / wall second"); plt.title(f"Independent environment throughput ({workers} worker processes)")
    for i,v in enumerate([count/serial_wall, count/parallel_wall]): plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(FIG / "parallel_throughput.png", dpi=160); plt.close()
    return out, serial, parallel


def main():
    for p in (FIG, VID, DATA): p.mkdir(parents=True, exist_ok=True)
    config = load_config()
    # A one-stroke run is the controlled single-stroke acceptance case.
    single_env, single_results, single_wall = run_demo(config, 101, [Stroke(-.06, -.04, .07, .012, .1, .60, 0)], "single_stroke")
    write_episode(DATA / "single_stroke_episode.json", single_env,
                  [Stroke(-.06, -.04, .07, .012, .1, .60, 0)], single_results)
    single_env.close()

    multi_strokes = flower_strokes() + fish_strokes()[:2]
    env, results, multi_wall = run_demo(config, 202, multi_strokes, "stage1")
    write_episode(DATA / "demo_episode.json", env, multi_strokes, results)
    env.close()

    # Re-open a final non-rendering copy for stable plots and exact data.
    plot_env = PaintingEnv(config, seed=202, render=True)
    plot_start = time.perf_counter()
    plot_results = [plot_env.execute_stroke(s) for s in multi_strokes]
    plot_wall = time.perf_counter() - plot_start
    stage_plots(config, plot_env, plot_results, multi_strokes)
    plot_env.close()

    throughput_metrics, serial, parallel = throughput(config)
    all_results = single_results + results
    contact = [r for r in all_results if r.get("force_acquired")]
    successful = [r for r in all_results if r.get("success")]
    metrics = {
        "scope": "Stage 1 environment validation; preset strokes only, no learned model or MPC planner",
        "seed": 202,
        "single_stroke": single_results[0],
        "multi_stroke_count": len(multi_strokes),
        "multi_success_count": int(sum(r.get("success", False) for r in results)),
        "all_stroke_count": len(all_results),
        "success_count": len(successful),
        "contact_acquisition_rate": len(contact)/len(all_results),
        "success_rate": len(successful)/len(all_results),
        "mean_force_N": float(np.mean([r["force_mean_N"] for r in successful])) if successful else 0.0,
        "mean_force_rmse_N": float(np.mean([r["force_rmse_N"] for r in successful])) if successful else 0.0,
        "mean_force_std_N": float(np.mean([r["force_std_N"] for r in successful])) if successful else 0.0,
        "mean_force_range_N": float(np.mean([r["force_range_N"] for r in successful])) if successful else 0.0,
        "max_force_N": float(np.max([r["force_max_N"] for r in successful])) if successful else 0.0,
        "mean_brush_tilt_deg": float(np.mean([r["brush_tilt_mean_deg"] for r in successful])) if successful else 0.0,
        "max_brush_tilt_deg": float(np.max([r["brush_tilt_max_deg"] for r in successful])) if successful else 0.0,
        "mean_trajectory_rmse_m": float(np.mean([r["trajectory_rmse_m"] for r in successful])) if successful else 0.0,
        "deposited_pixels": int(plot_env.canvas.changed_mask().sum()),
        "multi_sim_time_s": float(plot_env.data.time),
        "multi_wall_time_s_with_video": multi_wall,
        "simulation_realtime_factor_without_render": float(plot_env.data.time/max(plot_wall, 1e-9)),
        "minimum_stroke_contact_fraction": float(min(r['contact_fraction'] for r in successful)),
        "workstation_dip_mean_s": float(np.mean([r['duration_s'] for r in plot_env.workstation_results if r['phase'] == 'DIP'])),
        "workstation_dip_max_s": float(np.max([r['duration_s'] for r in plot_env.workstation_results if r['phase'] == 'DIP'])),
        "workstation_clean_mean_s": float(np.mean([r['duration_s'] for r in plot_env.workstation_results if r['phase'] == 'CLEAN'])),
        "workstation_clean_max_s": float(np.max([r['duration_s'] for r in plot_env.workstation_results if r['phase'] == 'CLEAN'])),
        "throughput": throughput_metrics,
        "mujoco_version": __import__("mujoco").__version__,
    }
    (REPORT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))

    report = f"""# IMPASTO 阶段 1：本地仿真环境验证报告

## 范围

本报告验证 `/home/lcy-pony/Draw_WS/IMPASTO` 中的阶段 1 环境：MuJoCo Panda、18×21 inch 画布、24 个颜料槽、清洗器、顶视相机、Bezier 笔画、接触力获取、蘸漆/清洗状态和并行仿真接口。演示笔画是固定预设轨迹，用于验收环境执行链路；阶段 2 learned pixel dynamics model、阶段 3 MPC 和论文 baseline 均未实现。

论文对齐部分包括 Panda 7 DoF、画布尺寸、24 色槽、顶视观测、六参数笔画 `(x0,y0,l,b,alpha,F)`、二次 Bezier 几何、Fig. A1 所示工具装配链以及“起笔建立法向力、随后保持高度完成笔画”的控制阶段结构。MuJoCo 刚体接触、50 mm 画笔的柔顺刷毛、油漆沉积、传感器读数和速度是本地仿真替代或工程选择，详见 `docs/fidelity.md`。

## 运行结果

| 指标 | 结果 |
|---|---:|
| 总笔画数（单笔+多笔） | {metrics['all_stroke_count']} |
| 成功笔画数 | {metrics['success_count']} |
| 接触力获取率 | {metrics['contact_acquisition_rate']:.3f} |
| 笔画成功率 | {metrics['success_rate']:.3f} |
| 各笔最低接触比例 | {metrics['minimum_stroke_contact_fraction']:.3f} |
| 平均取色 / 最大取色时间 | {metrics['workstation_dip_mean_s']:.2f} / {metrics['workstation_dip_max_s']:.2f} s |
| 平均清洗 / 最大清洗时间 | {metrics['workstation_clean_mean_s']:.2f} / {metrics['workstation_clean_max_s']:.2f} s |
| 平均刷尖/画布接触力 | {metrics['mean_force_N']:.3f} N |
| 平均法向力 RMSE | {metrics['mean_force_rmse_N']:.3f} N |
| 运笔阶段法向力标准差 | {metrics['mean_force_std_N']:.3f} N |
| 运笔阶段法向力范围 | {metrics['mean_force_range_N']:.3f} N |
| 最大法向力 | {metrics['max_force_N']:.3f} N |
| 平均接触轨迹 RMSE | {metrics['mean_trajectory_rmse_m']*1000:.2f} mm |
| 平均刷杆倾角 | {metrics['mean_brush_tilt_deg']:.3f}° |
| 最大刷杆倾角 | {metrics['max_brush_tilt_deg']:.3f}° |
| 最终画布沉积像素 | {metrics['deposited_pixels']} |
| 多笔物理仿真时间 | {metrics['multi_sim_time_s']:.3f} s |
| 多笔含渲染运行时间 | {metrics['multi_wall_time_s_with_video']:.3f} s |
| 关闭渲染时仿真/墙钟时间比 | {metrics['simulation_realtime_factor_without_render']:.2f}× |
| 串行吞吐量 | {throughput_metrics['serial_strokes_per_s']:.3f} stroke/s |
| {throughput_metrics['workers']} 进程并行吞吐量 | {throughput_metrics['parallel_strokes_per_s']:.3f} stroke/s |
| 并行加速比 | {throughput_metrics['speedup']:.2f}× |

单笔验收结果：`{single_results[0].get('reason')}`；多笔验收结果：{metrics['multi_success_count']}/{metrics['multi_stroke_count']} 成功。数据、逐步力曲线、末端轨迹和阶段状态转换均保存为 JSON/CSV。

## 媒体和数据

- [场景布局](figures/scene_layout.png)
- [画布前后对比](figures/canvas_before_after.png)
- [轨迹](figures/trajectory.png)
- [力跟踪](figures/force_tracking.png)
- [调色盘与清洗状态](figures/palette_and_cleaning.png)
- [并行吞吐量](figures/parallel_throughput.png)
- [单笔场景视频](videos/single_stroke_scene.mp4)
- [多笔场景视频](videos/stage1_scene.mp4)
- [多笔画布视频](videos/stage1_topdown.mp4)
- [多笔 episode](data/demo_episode.json)
- [单笔 episode](data/single_stroke_episode.json)
- [力曲线 CSV](data/force_trace.csv)
- [轨迹 CSV](data/trajectory.csv)
- [吞吐量 CSV](data/throughput.csv)

## 解释和限制

MuJoCo 的离散 penalty contact 在小刷尖上仍是接触模型，不是论文硬件的 Kunwei 六轴传感器；报告中的刷尖/画布法向力不再乘固定比例，`PaintingEnv.normal_force_raw()` 与之相同，工具安装点另提供 `tool_wrench()`。运笔中增加了低带宽的高度修正，以抵消离散接触和关节伺服误差；它用于阶段 1 仿真稳定性验证，不能等同于论文硬件上的六轴传感器力控。沉积由接触、力、速度、刷上颜料量和画布 tooth 驱动，不是 Navier–Stokes 或逐刷毛油画流体模型。视频中的预设笔画不代表 IMPASTO 学习模型已经生成了规划结果。

力控流程是先以 `force_ema` 滤波目标力与接触法向力误差，落笔阶段按误差调节高度；达到 ±0.04 N 并持续 0.16 s 后进入运笔。当前实现的内层是关节位置/姿态伺服，外层在每个控制步读取未缩放的 MuJoCo 接触法向力，并以低带宽高度修正近似实时导纳控制；它不是论文完整的 `J^T F_ff` 力矩注入实现。阶段 1 的 8 笔演示在有效接触样本上的平均力为 {metrics['mean_force_N']:.3f} N，目标力均值为 {float(np.mean([s.force for s in multi_strokes])):.3f} N；逐笔力标准差均值为 {metrics['mean_force_std_N']:.3f} N，逐笔力范围均值为 {metrics['mean_force_range_N']:.3f} N。结果表明本轮仿真绘制期间力基本稳定，但仍有接触模型与真实六轴传感器之间的系统差异。

画布状态不是在笔画结束后预先生成：每个控制步都使用实际刷尖位置、实际接触力、实际移动距离和当前载漆量更新；录像按 render_fps 对已更新的动态纹理采样。修正后的画布上传不再额外翻转图像，且关闭主光源投影到画布的阴影，使实际笔迹不会被机器人阴影遮挡。

下一阶段可以直接复用本阶段 episode 数据接口，采集论文对齐规模的 self-play 数据，加入 100×100 灰度局部状态、预渲染笔画和 6D action，训练 U-Net 前向动力学模型。
"""
    (REPORT / "report.md").write_text(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
