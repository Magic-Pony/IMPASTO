# 阶段 3：规划与闭环绘画

## 目标

用阶段二训练的 IMPASTO-UNet 对单笔和多笔绘画进行模型预测规划，并在 MuJoCo 环境中执行动作、读取实际画布后重新规划。本阶段不实现或比较论文 baseline。

## 论文协议

- 单笔任务使用 H=1 的规划窗口。
- 多笔任务使用 H=5，每次执行前两笔后观察画布并重新规划。
- Flower/Fish 各包含 17/18 笔；累计第 5、10、15 笔和最终图像定义每个规划阶段的目标。当前 refined 运行的最终配置为 Flower 17 笔、Fish 18 笔。
- 规划使用 PCA–RANSAC 笔画初始化、加权 L1 终端图像目标、elite softmax 的 MPPI 序列更新，以及按时间步更新协方差和步长的 CMA 风格自适应。
- 颜色从变化区域图像与本地调色板估计。当前正式结果使用 `reports/stage3_refined/color_database.json` 中由仿真采集的 25×25 颜料 patch 均值与协方差，并用 Mahalanobis 距离分类；几何更新后重新取色。论文的实测数据库未发布，因此这是本地工程替代物，并限制到 refined 阶段二已训练的 24 个本地预混颜色。

## 未公开信息与替代素材

论文未公布候选数、优化轮数、softmax 温度、RANSAC 阈值及完整 CMA 系数；本地值集中在 `configs/stage3_baseline_1107014_final.json`，不可视为论文原始超参数。作者绘制的 Flower/Fish 原始高分辨率图像和逐笔标注没有随论文发布。本地直接使用作者公开 Fig. 8 的 Flower/Fish 目标 panel 及其 2/5/7/10/12/15 笔检查点，动作仍由 PCA–RANSAC 和 learned-dynamics MPC 推断。

阶段二网络在灰度图上预测下一状态，所以规划的主目标也是灰度图；RGB 目标和画布只作为颜色选择和报告审计的辅助信号。当前 refined 配置使用 `opacity_scale=1.0` 的不透明任务颜料分支，并在训练、预测和 MuJoCo 执行中统一使用同一沉积模型。历史版本曾把照片中的透明度比直接乘到本地沉积率上，造成模型与执行器尺度不一致，该分支已从正式结果中移除。仿真接触和颜料沉积沿用阶段一的 MuJoCo penalty contact 与局部沉积模型，不代表油画颜料流变，也不等价于真实机器人。

当前正式配置是 `configs/stage3_baseline_1107014_final.json`：H=5、每轮执行 2 笔，64 个候选、36 次迭代、20% elite。`use_published_prefix_targets=false`；公开的 2/5/7/10/12/15 笔图像只用于审计和阶段目标，规划目标来自当前公开目标图和实时画布差分，不把不存在的逐笔 prefix 当作真实观测。

## 运行和产物

```bash
bash scripts/run.sh scripts/run_stage3.py
bash scripts/run.sh -m pytest -q
```

最终结果见 `reports/baseline_1107014_corrected_final/report.md` 和 `reports/final_reproduction/index.html`。`reports/stage3_refined/metrics.json` 保存运行设备、checkpoint 哈希、重规划周期、每笔动作和颜色、执行成功状态、接触力、轨迹误差、完整画布/目标区域误差及耗时。`reports/stage3_refined/flower_scene_replay.mp4`、`fish_scene_replay.mp4` 是从与保留基线相同的 MuJoCo 执行动作重放的连续场景视频，包含机器人、取色、清洗、画布、目标图和力读数；最终展示目录中的 `*_2x.mp4` 是它们的 2× 导出。旧的 `flower_painting.mp4`、`fish_painting.mp4` 只包含逐笔静态画布帧，不能替代场景重放。

LPIPS AlexNet 权重下载未通过 SHA-256 校验，本阶段没有计算 LPIPS。设置 `IMPASTO_ENABLE_LPIPS=1` 会尝试加载经验证的权重；不应使用随机初始化网络计算该指标。

当前最终数值、Flower 逐轮诊断、Fish 汇总和视觉限制见 [最终三阶段报告](../reports/final_reproduction/report.md)。
