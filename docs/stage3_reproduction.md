# 阶段 3：规划与闭环绘画

当前共享配置为 `configs/stage3_coverage_order.json`。选择性回退、重新数采训练、BN校准、通用RGB覆盖修复、新种子测试和完整命令见 [通用修正说明](general_refinement.md)。以下 refined 说明和数值对应历史版本；最终本轮结果见 [完整图文报告](../reports/refinement_coverage_order/index.html)。

## 目标

用阶段二训练的 IMPASTO-UNet 对单笔和多笔绘画进行模型预测规划，并在 MuJoCo 环境中执行动作、读取实际画布后重新规划。本阶段不实现或比较论文 baseline。

## 论文协议

- 单笔任务使用 H=1 的规划窗口。
- 多笔任务使用 H=5，每次执行前两笔后观察画布并重新规划。
- Flower/Fish 各包含 17/18 笔；累计第 5、10、15 笔和最终图像定义每个规划阶段的目标。当前 refined 运行的最终配置为 Flower 17 笔、Fish 18 笔。
- 规划使用 PCA–RANSAC 笔画初始化、加权 L1 终端图像目标、elite softmax 的 MPPI 序列更新，以及按时间步更新协方差和步长的 CMA 风格自适应。
- 颜色从变化区域图像与本地调色板估计。论文的颜色分类依赖 25×25 实测颜料 patch 数据库及 Mahalanobis 分类；该数据库未发布，所以本实现使用目标 RGB 与 palette 色差方向的最小残差，并限制到 refined 阶段二已训练的 24 个本地预混颜色。

## 未公开信息与替代素材

论文未公布候选数、优化轮数、softmax 温度、RANSAC 阈值及完整 CMA 系数；本地值集中在 `configs/stage3_refined.json`，不可视为论文原始超参数。作者绘制的 Flower/Fish 原始高分辨率图像和逐笔标注没有随论文发布。本地直接使用作者公开 Fig. 8 的 Flower/Fish 目标 panel 及其 2/5/7/10/12/15 笔检查点，动作仍由 PCA–RANSAC 和 learned-dynamics MPC 推断。

阶段二网络在灰度图上预测下一状态，所以规划的主目标也是灰度图；RGB 目标和画布只作为颜色选择和报告审计的辅助信号。当前 refined 配置使用 `opacity_scale=1.0` 的不透明任务颜料分支，并在训练、预测和 MuJoCo 执行中统一使用同一沉积模型。历史版本曾把照片中的透明度比直接乘到本地沉积率上，造成模型与执行器尺度不一致，该分支已从正式结果中移除。仿真接触和颜料沉积沿用阶段一的 MuJoCo penalty contact 与局部沉积模型，不代表油画颜料流变，也不等价于真实机器人。

当前正式配置是 `configs/stage3_refined.json`：H=5、每轮执行 2 笔，64 个候选、36 次迭代、20% elite。`use_published_prefix_targets=false`；公开的 2/5/7/10/12/15 笔图像只用于审计和阶段目标，规划目标来自当前公开目标图和实时画布差分，不把不存在的逐笔 prefix 当作真实观测。

## 运行和产物

```bash
bash scripts/run.sh scripts/run_stage3.py
bash scripts/run.sh -m pytest -q
```

最终结果见 `reports/refinement_final/report.md` 和 `reports/refinement_final/index.html`。`reports/stage3_refined/metrics.json` 保存运行设备、checkpoint 哈希、重规划周期、每笔动作和颜色、执行成功状态、接触力、轨迹误差、完整画布/目标区域误差及耗时。`flower_scene_replay.mp4`、`fish_scene_replay.mp4` 是从真实 MuJoCo 执行记录重放的连续场景视频，包含机器人、画布、目标图和力读数；旧的 `flower_painting.mp4`、`fish_painting.mp4` 只包含逐笔静态画布帧，不能替代场景重放。

LPIPS AlexNet 权重下载未通过 SHA-256 校验，本阶段没有计算 LPIPS。设置 `IMPASTO_ENABLE_LPIPS=1` 会尝试加载经验证的权重；不应使用随机初始化网络计算该指标。

当前最终数值、Flower 逐轮诊断、Fish 汇总和视觉限制见 [最终优化报告](../reports/refinement_final/report.md)。
