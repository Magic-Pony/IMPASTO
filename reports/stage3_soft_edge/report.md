# IMPASTO 阶段三：学习模型驱动的笔画规划

## 结果

规划器使用阶段二训练的 IMPASTO-UNet、PCA–RANSAC 初始化、MPPI elite softmax 序列更新及 CMA 风格自适应。多笔绘画采用 H=5，每轮最多执行 2 笔后读取仿真画布并重新规划。本配置的规划目标为加权灰度 L1 + 1.0 × RGB 辅助损失，后者为本地扩展；不将综合规划代价标作论文的灰度 L1。

|任务|目标笔数|执行笔数|综合规划代价均值|执行加权灰度 L1|LPIPS AlexNet|
|---|---:|---:|---:|---:|---|
|single_stroke|1|1|0.00183|0.00121|unavailable|
|flower|17|17|0.02205|0.01537|unavailable|
|fish|18|18|0.02601|0.01529|unavailable|

规划代价是各重规划周期终端预测代价的均值；执行 L1 是最终真实仿真画布与目标图的误差，两列含义不同。兼容旧日志而保留的 `planning_weighted_gray_l1` 字段在启用 RGB 辅助项时也是综合代价。完整全画布 L1、改变区域 L1、逐周期代价、每笔力控及轨迹数据见 `metrics.json`。本阶段没有运行或比较任何 baseline。LPIPS could not be initialized: `disabled by default; set IMPASTO_ENABLE_LPIPS=1 after verifying AlexNet weights; a previous download did not pass SHA-256 validation`

## 执行质量

|任务|全部执行成功|法向力 RMSE (N)|轨迹 RMSE (m)|
|---|---:|---:|---:|
|single_stroke|1|0.01008|0.000066|
|flower|1|0.00717|0.000115|
|fish|1|0.01842|0.000123|

## 模型预测与实际执行

|任务|执行相同动作后的全画布预测 L1|目标差异区预测 L1|平均透明度 α|
|---|---:|---:|---:|
|flower|0.00044|0.00961|0.956|
|fish|0.00037|0.01267|0.853|

此处逐轮比较模型对已执行动作的预测与同一轮真实仿真观测。

## 任务结果

### single_stroke

![single_stroke result](single_result.png)

[Target](single_target.png) · [Execution video](single_stroke.mp4)

### flower

![flower result](flower_result.png)

[Target](flower_target.png) · [Execution video](flower_painting.mp4)

### fish

![fish result](fish_result.png)

[Target](fish_target.png) · [Execution video](fish_painting.mp4)


## 论文对齐与边界

- 单笔：H=1，以目标差分初始化并优化笔画参数，再执行该动作并读取真实仿真画布。当前 `single_stroke` 是本地生成的诊断曲线，不能代替论文五位画家的人类笔迹实验。
- Flower/Fish：保留论文所述 17/18 笔目标规模；目标图和 2/5/7/10/12/15 笔公开检查点来自作者 Fig. 8，使用目标笔数 5、10、15 和最终状态作为窗口目标，H=5，每轮执行两笔并闭环重规划。
- 论文使用的 H=5、每轮两笔及 PCA–RANSAC/MPPI/CMA 规划结构已落实。论文未公开候选数、迭代数、softmax 温度、RANSAC 阈值和 CMA 系数；本地取值见 `configs/stage3_refined.json`。
- 论文没有随文章发布 artist Flower/Fish 的原始高分辨率文件和逐笔颜色/动作。这里使用作者公开 Fig. 8 的低分辨率目标 panel，保留目标笔画数并由规划器推断动作，不冒充原始逐笔标注或定量复刻。
    - 修订配置使用仿真执行采集的 25×25 颜料块均值、协方差和 Mahalanobis 分类，几何改变后重新取色。每色两种力，协方差加正则项；这是本地数据库，不是作者的实物数据库。中心带采样近似论文中心圆盘与预测足迹交集。
    - 历史配置曾把附录 B 的分类比值作为连续透明度，并乘本地系数；该做法与真实沉积不等价，修订配置仅使用不透明分支。当前 `opacity_scale=1.000`；不能据此声称实现了论文完整透明度模块。
- 本地阶段一使用 MuJoCo penalty contact 与简化颜料沉积，故执行结果是仿真验证，不等价于真实油画物理或论文硬件试验。
- LPIPS AlexNet 未计算：torchvision 预训练 AlexNet 权重下载未通过 SHA-256 校验；不会以随机网络给出感知指标。

## 可复现文件

- 规划器配置：`configs/stage3_refined.json`
- 聚合数值和逐笔动作/力控日志：`reports/stage3_soft_edge/metrics.json`
- Stage-2 checkpoint SHA-256：`39b262e9c481e3d1a7840d4fafa10726bab72c810960df0e218c0faaa3ab42ff`
- 运行命令：`IMPASTO_STAGE3_CONFIG=configs/stage3_refined.json bash scripts/run.sh scripts/run_stage3.py`

## 逐轮诊断图

### flower

![flower_cycle_01](flower_cycle_01.png)

![flower_cycle_02](flower_cycle_02.png)

![flower_cycle_03](flower_cycle_03.png)

![flower_cycle_04](flower_cycle_04.png)

![flower_cycle_05](flower_cycle_05.png)

![flower_cycle_06](flower_cycle_06.png)

![flower_cycle_07](flower_cycle_07.png)

![flower_cycle_08](flower_cycle_08.png)

![flower_cycle_09](flower_cycle_09.png)

![flower_cycle_10](flower_cycle_10.png)
### fish

![fish_cycle_01](fish_cycle_01.png)

![fish_cycle_02](fish_cycle_02.png)

![fish_cycle_03](fish_cycle_03.png)

![fish_cycle_04](fish_cycle_04.png)

![fish_cycle_05](fish_cycle_05.png)

![fish_cycle_06](fish_cycle_06.png)

![fish_cycle_07](fish_cycle_07.png)

![fish_cycle_08](fish_cycle_08.png)

![fish_cycle_09](fish_cycle_09.png)

![fish_cycle_10](fish_cycle_10.png)

![fish_cycle_11](fish_cycle_11.png)
