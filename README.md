# IMPASTO 本地仿真复现

复现对象：*IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction*，ICRA 2026，arXiv:2603.29315v1。

所有代码、原文、依赖环境、缓存、数据和报告保存在本目录。Git 跟踪源代码、论文、配置和阶段报告；独立 Conda 前缀为 `.conda/`，缓存和大批量采集数据不进入 Git。当前工作为阶段 1；阶段 2/3 的学习和逆向规划尚未开展。

## 环境

```bash
conda env create --prefix ./.conda --file environment.yml
```

激活：`conda activate /home/lcy-pony/Draw_WS/IMPASTO/.conda`。
推荐通过 `bash scripts/run.sh ...` 运行，避免本机 ROS 的 `PYTHONPATH` 污染独立环境。

## 文档与结果

- [四阶段计划](docs/reproduction_plan.md)
- [论文一致性和替代假设](docs/fidelity.md)
- [数据接口](docs/data_contract.md)
- `papers/`：正文、附录、文本提取和来源清单。
- `assets/`：机器人资源、许可证及场景。
- `configs/`：仿真与实验配置。
- `src/impasto/`：环境、控制、沉积、数据采集及后续模型/规划接口。
- `tests/`：动力学、接触与数据可重复性测试。
- `reports/stage1/`：实际执行生成的文字、图片、视频、数值报告。

首次公开代码检索时，作者项目页的代码入口为 Code coming soon。本项目是依据论文的独立实现，不能宣称使用作者原始代码、真实自探索数据或忠实油画流体模拟器。
