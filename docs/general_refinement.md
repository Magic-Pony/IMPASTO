# 1107014 选择性回退后的通用修正

本轮以 `1107014` 的对称灰度目标、RGB 辅助权重 1、原有宽度与沉积律为基础。旧结果和失败试验保留；没有重置历史、改写目标图、插入目标动作，也没有按 Flower/Fish 选择颜色子集、限制力或设置专用坐标。最终完整证据入口是 [通用修正报告](../reports/refinement_coverage_order/index.html)，逐轮文字在 [report.md](../reports/refinement_coverage_order/report.md)。此前的 `refinement_final` 是历史结果。

## 修改边界

- **物理**：`stage1_stable.json` 将刷尖 MuJoCo 接触松弛时间从 6ms 改成 20ms，保留 2ms 积分步长、4ms 控制周期、持续力反馈、刷毛弹簧及沉积律。它修复本地低力振荡，不代表论文控制器公式错误。
- **数据与训练**：`stage2_stable.json` 收集 2000 个随机转移，20个episode各100笔，1800训练/200验证。低力分层和质量拒收提高细笔数据覆盖。UNet、Adam、batch120、1000epochs、学习率衰减保持原设计。数据预算高于论文900/100，属于通用扩展。
- **部署统计**：新模型原始输出在变化区域退化。`audit_model_calibration.py` 仅用未增强训练集、按前向顺序重估 BatchNorm 总体统计，不改变权重；版本选择使用验证集，因此这不是独立测试。
- **候选与颜色**：保留薄线、避免RGB区域被二次灰度门限丢弃；PCA/RANSAC增补颜色连通区域及多力假设，使用模型排序。颜色采样实现足迹与中心圆盘交集、距离加权、线性RGB均值、Mahalanobis分类；数据库仍来自仿真而非作者实测数据。
- **RGB覆盖**：灰度接近时原转换无法识别不同颜料的覆盖。新转换查询同一网络在空白参考上的预测，估计覆盖率并用于歧义区域；不是从目标图或真实未来得到覆盖率。
- **规划**：H≤5枚举种子顺序，再使用正常MPPI/CMA更新。预测灰度按本地凸混合沉积范围投影。两者是通用扩展，不能标为论文原算法。
- **速度**：disk stamp预渲染改成最近采样点查询，200随机动作逐像素相同，局部函数快3.10倍；未改变64候选×36迭代的艺术目标预算。推理batch8用于控制共享GPU显存。

## 复现命令

所有命令在仓库根目录通过独立 `.conda` 环境执行。以下数采/训练/校准会更新现有数据或模型，完整归档已经保存在本机，不需要为了查看报告重跑。

```bash
IMPASTO_STAGE2_CONFIG=configs/stage2_stable.json bash scripts/run.sh scripts/collect_selfplay.py
IMPASTO_STAGE2_CONFIG=configs/stage2_stable.json bash scripts/run.sh scripts/train_dynamics.py
IMPASTO_STAGE2_CONFIG=configs/stage2_stable.json bash scripts/run.sh scripts/evaluate_dynamics.py --comparison-checkpoint outputs/stage2_refined/impasto_unet_best.pt
bash scripts/run.sh scripts/audit_model_calibration.py
```

最终完整绘画与审计：

```bash
IMPASTO_STAGE3_CONFIG=configs/stage3_coverage_order.json bash scripts/run.sh scripts/run_stage3.py --report-dir reports/refinement_coverage_order
bash scripts/run.sh scripts/audit_refinement.py --report-dir reports/refinement_coverage_order --out-dir reports/refinement_coverage_order/audit --previous-report reports/stage3_rollback_1107014 --rgb-prediction
bash scripts/run.sh scripts/replay_scene_video.py --report-dir reports/refinement_coverage_order --task flower --fps 3
bash scripts/run.sh scripts/replay_scene_video.py --report-dir reports/refinement_coverage_order --task fish --fps 3
bash scripts/run.sh scripts/build_general_refinement_report.py
```

报告生成器会从两段已完成MP4按相同相对进度抽取六帧，保存contact sheet及`video_review.json`。报告生成还需要 `visual_notes.json`：必须人工逐轮检查后写入，不能由误差数字自动代替视觉评价。每轮有八图诊断、RGB终态/执行前缀对照及每笔预测。连续视频由保存的动作重新执行，逐像素核对最终画布，含清洗、取色、接触和实时沉积。

通用验证：

```bash
IMPASTO_STAGE3_CONFIG=configs/stage3_coverage_order.json bash scripts/run.sh scripts/validate_general_planning.py --samples 8 --strokes 3 --out-dir reports/general_validation_coverage_order
IMPASTO_STAGE3_CONFIG=configs/stage3_coverage_order.json bash scripts/run.sh scripts/validate_general_planning.py --seed-start 2026101500 --samples 8 --strokes 4 --out-dir reports/general_validation_fresh
IMPASTO_STAGE3_CONFIG=configs/stage3_coverage_order.json bash scripts/run.sh scripts/validate_chromatic_overlap.py --out-dir reports/chromatic_overlap_audit
IMPASTO_STAGE3_CONFIG=configs/stage3_calibrated.json bash scripts/run.sh scripts/validate_force_model.py --out-dir reports/force_model_calibrated
bash scripts/run.sh -m pytest -q
```

开发目标从2026101100开始，反复用于方案选择；不能称为独立测试。新种子目标从2026101500开始，在方案冻结后使用，未将生成它们的动作提供给规划器。通用测试的32候选×16迭代预算对旧新版本一致；低于Flower/Fish预算，数值只能按其自身协议解读。绘画目标是公开Fig.8裁剪，检查点遵循论文第5/10/15/final图像；真实观测始终来自MuJoCo执行。

## 证据位置和未解决问题

- `reports/stage2_stable`：2000条采集质量、1000epochs训练、原始模型验证、数据清单和哈希。
- `reports/model_calibration_audit`：6个原始/校准模型版本在同一验证集的结果。
- `reports/force_model_stable_old`、`force_model_stable_new`、`force_model_calibrated`：力稳定性与预测误差分开测量。
- `reports/chromatic_overlap_audit`：8组等灰度异色交叉笔迹；RGB修复的独立功能证据。
- `reports/general_validation_coverage_order`：开发对照；`general_validation_fresh`：新种子测试。
- `reports/refinement_calibrated`：无覆盖率/顺序搜索的完整绘画对照。
- `reports/refinement_coverage_order`：最终共享配置的完整绘画、逐轮视觉审计和场景视频。
- `reports/refinement_stable/attempt_status.json`：原始新模型绘画中止记录，不是完整最终结果。

油漆模型仍是接触驱动的局部混合，没有刷毛分辨形变、真实油漆流体和传感器噪声。恒力、恒色二次曲线难以表达画家笔迹的变宽、尖端和浓淡。新校准模型在部分独立局部预测测试中仍弱于旧模型；训练和随机目标改善不能保证任意艺术照片都能复现。最终报告如实区分这些限制与已修复问题，没有计算不可用的LPIPS或复现论文baselines。

本轮整幅灰度误差Flower下降5.61%、Fish下降8.52%，但Fish前景IoU由0.7279降至0.6208、召回由0.7664降至0.6351；单笔L1退化20.65%。最终候选归档供审查，旧版本保留，不能称画质全面优于1107014。
