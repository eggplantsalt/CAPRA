#!/usr/bin/env bash
# =============================================================
# smoke_capra.sh -- CAPRA 纯 Python 逻辑冒烟测试
# =============================================================
#
# 作用
# ----
# 验证 CAPRA 的所有 Python 模块能否被正确导入和运行。
# 不需要 GPU、不需要模型检查点、不需要 LIBERO 安装。
# 是安装环境后的第一步验证。
#
# 覆盖范围（11 个检查组）
# ----------------------
#   [1]  CAPRAConfig 和 FinetuneCAPRAConfig 数据类
#   [2]  任务等价集过滤逻辑
#   [3]  安全目标分布 q_hat_t 构建
#   [4]  CAPRA KL 损失计算（CPU 张量）
#   [5]  训练分支：Baseline / CAPRA / 预热阶段
#   [6]  SPIR / EAR 指标计算
#   [7]  EpisodeMetrics + AggregateMetrics 聚合
#   [8]  报告生成（JSON + CSV + Markdown）
#   [9]  前驱归因权重 w_t 计算
#   [10] 程序化场景模板（4 种，使用 mock env）
#   [11] CAPRAEnvAdapter no-sim 路径
#
# 用法
# ----
#   bash scripts/capra/smoke_capra.sh
#
# 预期输出（成功）
# ---------------
#   [smoke_capra] All checks passed.
#
# 如果失败
# --------
#   直接运行 Python 脚本查看详细错误栈：
#   python scripts/capra/_smoke_logic.py
# =============================================================

set -euo pipefail

# 自动定位项目根目录（无论从哪里调用此脚本都能正确找到）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[smoke_capra] 项目根目录: $REPO_ROOT"
echo "[smoke_capra] 开始纯 Python 逻辑检查..."

# 运行实际的检查逻辑（在 _smoke_logic.py 中实现，保持 sh 文件简洁）
python scripts/capra/_smoke_logic.py

echo "[smoke_capra] 所有检查通过 / All checks passed."
