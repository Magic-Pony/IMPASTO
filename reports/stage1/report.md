# IMPASTO 阶段 1：本地仿真环境验证报告

## 范围

本报告验证 `/home/lcy-pony/Draw_WS/IMPASTO` 中的阶段 1 环境：MuJoCo Panda、18×21 inch 画布、24 个颜料槽、清洗器、顶视相机、Bezier 笔画、接触力获取、蘸漆/清洗状态和并行仿真接口。演示笔画是固定预设轨迹，用于验收环境执行链路；阶段 2 learned pixel dynamics model、阶段 3 MPC 和论文 baseline 均未实现。

论文对齐部分包括 Panda 7 DoF、画布尺寸、24 色槽、顶视观测、六参数笔画 `(x0,y0,l,b,alpha,F)`、二次 Bezier 几何、Fig. A1 所示工具装配链以及“起笔建立法向力、随后保持高度完成笔画”的控制阶段结构。MuJoCo 刚体接触、50 mm 画笔的柔顺刷毛、油漆沉积、传感器读数和速度是本地仿真替代或工程选择，详见 `docs/fidelity.md`。

## 运行结果

| 指标 | 结果 |
|---|---:|
| 总笔画数（单笔+多笔） | 9 |
| 成功笔画数 | 9 |
| 接触力获取率 | 1.000 |
| 笔画成功率 | 1.000 |
| 各笔最低接触比例 | 1.000 |
| 平均刷尖/画布接触力 | 0.584 N |
| 平均法向力 RMSE | 0.022 N |
| 运笔阶段法向力标准差 | 0.011 N |
| 运笔阶段法向力范围 | 0.053 N |
| 最大法向力 | 0.641 N |
| 平均接触轨迹 RMSE | 2.66 mm |
| 平均刷杆倾角 | 0.008° |
| 最大刷杆倾角 | 0.027° |
| 最终画布沉积像素 | 7585 |
| 多笔物理仿真时间 | 178.416 s |
| 多笔含渲染运行时间 | 282.859 s |
| 关闭渲染时仿真/墙钟时间比 | 9.58× |
| 串行吞吐量 | 0.361 stroke/s |
| 4 进程并行吞吐量 | 1.318 stroke/s |
| 并行加速比 | 3.65× |

单笔验收结果：`ok`；多笔验收结果：8/8 成功。数据、逐步力曲线、末端轨迹和阶段状态转换均保存为 JSON/CSV。

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

力控流程是先以 `force_ema` 滤波目标力与接触法向力误差，落笔阶段按误差调节高度；达到 ±0.04 N 并持续 0.16 s 后进入运笔。当前实现的内层是关节位置/姿态伺服，外层在每个控制步读取未缩放的 MuJoCo 接触法向力，并以低带宽高度修正近似实时导纳控制；它不是论文完整的 `J^T F_ff` 力矩注入实现。阶段 1 的 8 笔演示在有效接触样本上的平均力为 0.584 N，目标力均值为 0.581 N；逐笔力标准差均值为 0.011 N，逐笔力范围均值为 0.053 N。结果表明本轮仿真绘制期间力基本稳定，但仍有接触模型与真实六轴传感器之间的系统差异。

画布状态不是在笔画结束后预先生成：每个控制步都使用实际刷尖位置、实际接触力、实际移动距离和当前载漆量更新；录像按 render_fps 对已更新的动态纹理采样。修正后的画布上传不再额外翻转图像，且关闭主光源投影到画布的阴影，使实际笔迹不会被机器人阴影遮挡。

下一阶段可以直接复用本阶段 episode 数据接口，采集论文对齐规模的 self-play 数据，加入 100×100 灰度局部状态、预渲染笔画和 6D action，训练 U-Net 前向动力学模型。
