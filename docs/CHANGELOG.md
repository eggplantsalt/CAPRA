# CAPRA 工程变更日志 (CHANGELOG.md)

> 本文件记录所有对文档、目录结构的重组与删除决策。
> 代码逻辑变更请查阅 `docs/progress_capra.md`。

---

## 2026-03-15  Step 1&2：代码目录解耦 + 文档梳理

### 新增目录结构

`experiments/robot/capra/` 下按功能拆分为四个子包：

```
experiments/robot/capra/
  core/       配置、损失函数、等价集、足迹、状态信号
  scene/      对象角色、任务进度、前驱归因、数据集构建
  mining/     挖掘主循环、缓存管理、候选动作、快照、环境适配器、rollout
  eval/       评估循环、安全指标、报告生成、程序化场景模板
```

### 文件迁移清单

| 原路径 | 新路径 | 说明 |
|---|---|---|
| `capra/capra_config.py` | `capra/core/capra_config.py` | 配置中心 |
| `capra/capra_loss.py` | `capra/core/capra_loss.py` | CAPRA 损失函数 |
| `capra/equivalence.py` | `capra/core/equivalence.py` | 任务等价集 |
| `capra/footprint.py` | `capra/core/footprint.py` | 足迹函数 |
| `capra/signals.py` | `capra/core/signals.py` | 状态信号提取 |
| `capra/state_api.py` | `capra/core/state_api.py` | 信号 re-export shim |
| `capra/object_roles.py` | `capra/scene/object_roles.py` | 对象角色分类 |
| `capra/task_progress.py` | `capra/scene/task_progress.py` | 任务进度估计 |
| `capra/precursor.py` | `capra/scene/precursor.py` | 前驱归因 |
| `capra/build_capra_dataset.py` | `capra/scene/build_capra_dataset.py` | 数据集构建 |
| `capra/run_capra_mining.py` | `capra/mining/run_capra_mining.py` | 挖掘入口 |
| `capra/mining_cache.py` | `capra/mining/mining_cache.py` | 挖掘缓存管理 |
| `capra/buffer.py` | `capra/mining/buffer.py` | 训练样本读取器 |
| `capra/rollout.py` | `capra/mining/rollout.py` | 短时 CF rollout |
| `capra/candidate_actions.py` | `capra/mining/candidate_actions.py` | 候选动作采样 |
| `capra/snapshot.py` | `capra/mining/snapshot.py` | MuJoCo 快照/还原 |
| `capra/env_adapter.py` | `capra/mining/env_adapter.py` | 环境适配器 |
| `capra/run_capra_eval.py` | `capra/eval/run_capra_eval.py` | 评估入口 |
| `capra/metrics.py` | `capra/eval/metrics.py` | 安全指标 |
| `capra/report_utils.py` | `capra/eval/report_utils.py` | 报告生成器 |
| `capra/procedural_splits.py` | `capra/eval/procedural_splits.py` | 程序化场景模板 |

### 向后兼容 shim

原路径下的每个文件已替换为 shim 文件（`from new_path import *`），
确保使用旧导入路径的代码仍能正常工作。

例：`from experiments.robot.capra.metrics import SPIR` 仍然有效。

### 脚本重组

`scripts/capra/` 下新增 `_smoke_logic.py`（被 `smoke_capra.sh` 调用）。
原有 4 个 shell 脚本保留，`eval_capra.sh` 新增第 5 个位置参数 `side_effect_template`。

### 中文注释添加

以下文件添加了详细中文注释块（以 `#` 注释行形式，不破坏原有 docstring）：

| 文件 | 注释内容摘要 |
|---|---|
| `core/capra_config.py` | 全字段中文说明，含 shuffle_buffer_size 警告 |
| `core/capra_loss.py` | 核心公式、KL 方向、连续动作近似原理 |
| `core/equivalence.py` | 等价集过滤逻辑、双阈值设计理由 |
| `core/footprint.py` | 三分量分解、任务条件化规则 |
| `core/signals.py` | 信号精度说明、优雅降级策略 |
| `eval/metrics.py` | 全部指标物理含义、Baseline 模式说明 |
| `eval/report_utils.py` | 输出文件格式说明 |
| `mining/run_capra_mining.py` | 数据流、断点续传机制 |
| `vla-scripts/finetune_capra.py` | 两种模式说明、关键参数警告、WandB 指标解读 |

---

## 文档目录规划（新增）

当前 `docs/` 目录结构：

```
docs/
  CAPRA.md            算法规格说明书（冻结，禁止修改）
  CONTEXT.md          全局工程约束（冻结，禁止修改）
  README_CAPRA.md     项目 README（英文，工程师快速参考）
  CHANGELOG.md        本文件，变更记录
  progress_capra.md   各阶段实现进度日志
  capra_state.json    机器可读的阶段状态
  capra_plan.md       原始规划文档
```

计划在 Step 3 新增（中文文档体系）：

```
docs/zh/
  00_目录指引.md          项目文件树 + 核心文件位置说明
  01_快速开始.md           环境安装 + 最短路径验证
  02_数据集准备.md         RLDS 格式 + 路径规则 + 字段说明
  03_训练指南.md           Baseline + CAPRA 训练命令 + 参数解读
  04_评估指南.md           评估脚本 + 指标解读 + 结果分析
  05_程序化场景.md         四种 side-effect 模板使用说明
  06_故障排查.md           OOM/路径/显存/挂起等常见问题
  07_论文解读.md           代码背后的研究动机与方法解读
```

---

## 文档保留/删除决策

| 文件 | 决策 | 理由 |
|---|---|---|
| `docs/CAPRA.md` | **保留（禁止修改）** | 算法规格，冻结参考 |
| `docs/CONTEXT.md` | **保留（禁止修改）** | 全局约束，冻结参考 |
| `docs/progress_capra.md` | **保留** | 完整的阶段实现日志，对接手工程师有价值 |
| `docs/capra_state.json` | **保留** | 机器可读状态，脚本可解析 |
| `docs/capra_plan.md` | **保留** | 原始设计意图，供对比参考 |
| `docs/README_CAPRA.md` | **保留+更新** | 工程快速参考，需更新新目录结构 |
| `docs/CHANGELOG.md` | **新增** | 本文件 |
