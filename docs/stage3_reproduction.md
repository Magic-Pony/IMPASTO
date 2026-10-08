# 阶段 3：规划与闭环绘画

## 目标

用阶段二训练的 IMPASTO-UNet 对单笔和多笔绘画进行模型预测规划，并在 MuJoCo 环境中执行动作、读取实际画布后重新规划。本阶段不实现或比较论文 baseline。

## 论文协议

- 单笔任务使用 H=1 的规划窗口。
- 多笔任务使用 H=5，每次执行前两笔后观察画布并重新规划。
- Flower/Fish 各包含 17/18 笔；累计第 5、10、15 笔和最终图像定义每个规划阶段的目标。
- 规划使用 PCA–RANSAC 笔画初始化、加权 L1 终端图像目标、elite softmax 的 MPPI 序列更新，以及按时间步更新协方差和步长的 CMA 风格自适应。
- 颜色从变化区域图像与本地调色板估计。论文的颜色分类依赖 25×25 实测颜料 patch 数据库及 Mahalanobis 分类；该数据库未发布，所以本实现使用目标 RGB 与 palette 色差方向的最小残差，并限制到阶段二已训练的八种颜色。

## 未公开信息与替代素材

论文未公布候选数、优化轮数、softmax 温度、RANSAC 阈值及完整 CMA 系数；本地值集中在 `configs/stage3.json`，不可视为论文原始超参数。作者绘制的 Flower/Fish 原始高分辨率图像和逐笔标注没有随论文发布。本地直接使用作者公开 Fig. 8 的 Flower/Fish 目标 panel 及其 2/5/7/10/12/15 笔检查点，动作仍由 PCA–RANSAC 和 learned-dynamics MPC 推断。

阶段二网络在灰度图上预测下一状态，所以规划目标也是灰度图。执行结果仍保存 RGB。附录 B 的透明度比来自相机照片，而本地 Canvas 的 opacity 直接乘在满载笔刷沉积率上，因此阶段三使用 `opacity_scale=0.2` 做仿真颜料/相机标定；预测和 MuJoCo 执行使用同一有效透明度。仿真接触和颜料沉积沿用阶段一的 MuJoCo penalty contact 与局部沉积模型，不代表油画颜料流变，也不等价于真实机器人。

## 运行和产物

```bash
bash scripts/run.sh scripts/run_stage3.py
bash scripts/run.sh -m pytest -q
```

`reports/stage3/report.md` 汇总规划与执行误差并嵌入目标/结果图及视频链接。`metrics.json` 保存运行设备、checkpoint 哈希、重规划周期、每笔动作和颜色、执行成功状态、接触力、轨迹误差、完整画布/目标区域误差及耗时。`single_stroke.mp4`、`flower_painting.mp4`、`fish_painting.mp4` 为执行过程视频。

LPIPS AlexNet 权重下载未通过 SHA-256 校验，本阶段没有计算 LPIPS。设置 `IMPASTO_ENABLE_LPIPS=1` 会尝试加载经验证的权重；不应使用随机初始化网络计算该指标。
