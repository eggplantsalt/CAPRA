import pathlib

p = pathlib.Path(r'E:/CAPRA/docs/zh/08_实验操作完整指南.md')
existing = p.read_text(encoding='utf-8')

addition = """

### 评估全部 4 个套件

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    bash scripts/capra/eval_capra.sh \\
        tmp/models/openvla-7b-oft-finetuned-libero-${SUITE} ${SUITE} 50
done
```

### 查看结果

```bash
python -c "
import json, glob
f = sorted(glob.glob('experiments/logs/capra/CAPRA-EVAL-libero_spatial-*/results_aggregate.json'))
if f:
    d = json.load(open(f[-1]))
    print('success_rate:', round(d['success_rate'],3))
    print('topple_rate: ', d.get('topple_rate','N/A'))
"
```

`spir_mean`/`ear_mean` 此时为 0 是正确空值（obs-only 模式不计算这两个指标）。

---

## 11. 实验 B：Baseline 训练

**目的：** 用 L1 回归微调 OpenVLA-7B 作为 CAPRA 的公平对照组。只评估预训练检查点可**跳过**。

**前提：** `tmp/models/openvla-7b` 和 `tmp/datasets/rlds/libero_spatial_no_noops` 已准备好。

```bash
conda activate openvla && cd $PROJECT_ROOT

torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
    vla-scripts/finetune_capra.py \\
    --vla_path tmp/models/openvla-7b \\
    --dataset_name libero_spatial_no_noops \\
    --data_root_dir tmp/datasets/rlds \\
    --run_root_dir runs \\
    --capra_enabled False \\
    --use_l1_regression True --use_proprio True \\
    --num_images_in_input 2 \\
    --use_lora True --lora_rank 32 \\
    --batch_size 8 --shuffle_buffer_size 2000 \\
    --learning_rate 5e-4 \\
    --num_steps_before_decay 100000 --max_steps 200000 \\
    --save_freq 10000 --image_aug True
# 多卡：把 --nproc-per-node 1 改为实际 GPU 数量（如 4）
```

**健康指标：** L1 loss 降至 0.01 以下（约 150K 步）；LIBERO-Goal 例外，50K 步通常更好。

---

## 12. 实验 C：CAPRA 离线挖掘

**目的：** 用基线模型在 LIBERO 仿真中运行，对每步采样 K 个候选动作，通过短时反事实 rollout 计算 P_t/F_t，生成安全监督信号缓存。

**前提：** 第 7 步模型已下载；LIBERO 已安装；服务器支持 MuJoCo 渲染。

### 设置 MuJoCo 离屏渲染（无显示器服务器必须）

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc
# 需 root 安装：sudo apt-get install -y libosmesa6-dev
# 无 root 时试：export MUJOCO_GL=egl
```

### 运行挖掘

```bash
conda activate openvla && cd $PROJECT_ROOT

# 一键脚本（推荐）：参数 = 检查点 / 数据集名 / 缓存目录
bash scripts/capra/mine_capra.sh \\
    tmp/models/openvla-7b-oft-finetuned-libero-spatial \\
    libero_spatial tmp/capra_cache

# 完整 CLI（更多控制）：
python -m experiments.robot.capra.mining.run_capra_mining \\
    --pretrained_checkpoint tmp/models/openvla-7b-oft-finetuned-libero-spatial \\
    --dataset_name libero_spatial --cache_root tmp/capra_cache \\
    --num_mining_episodes 50 --K 8 --H_s 5 --W 10 \\
    --progress_floor 0.20 --epsilon_p_abs 0.05 --seed 7

# 快速验证（1 个 episode，约 10-30 分钟）：
python -m experiments.robot.capra.mining.run_capra_mining \\
    --pretrained_checkpoint tmp/models/openvla-7b-oft-finetuned-libero-spatial \\
    --dataset_name libero_spatial --cache_root tmp/capra_cache \\
    --num_mining_episodes 1
```

| 参数 | 默认值 | 调整建议 |
|---|---|---|
| `--K` | 8 | 减小(4)加速；增大(16)提升质量 |
| `--H_s` | 5 | 减小(3)加速；增大(8)提升精度 |
| `--num_mining_episodes` | 50 | 先用 1 验证流程 |
| `--progress_floor` | 0.20 | 激活率为 0 时降至 0.10 |

**断点续传：** 中途中断后直接重新运行，已完成的 episode 自动跳过。

### 验证

```bash
ls tmp/capra_cache/libero_spatial/ | wc -l
# 应等于 num_mining_episodes（默认 50）
```

---

## 13. 实验 D：CAPRA 训练

**目的：** 用 CAPRA 损失微调 OpenVLA-7B，得到具有内生安全偏好的 VLA 模型。
总损失：`L = L_anchor + lam * L_capra`

**前提：** 实验 C 缓存已生成；`tmp/models/openvla-7b` 和数据集已下载。

```bash
conda activate openvla && cd $PROJECT_ROOT

# 一键脚本：参数 = 起始模型 / 数据集（含_no_noops）/ 缓存目录
bash scripts/capra/train_capra.sh \\
    tmp/models/openvla-7b libero_spatial_no_noops tmp/capra_cache

# 完整 CLI：
torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
    vla-scripts/finetune_capra.py \\
    --vla_path tmp/models/openvla-7b \\
    --dataset_name libero_spatial_no_noops \\
    --data_root_dir tmp/datasets/rlds \\
    --cache_root tmp/capra_cache \\
    --run_root_dir runs \\
    --capra_enabled True \\
    --lam 0.1 --rho 0.5 --beta 1.0 \\
    --capra_warmup_steps 500 \\
    --use_l1_regression True --use_proprio True \\
    --num_images_in_input 2 \\
    --use_lora True --lora_rank 32 \\
    --batch_size 8 --shuffle_buffer_size 2000 \\
    --learning_rate 5e-4 \\
    --num_steps_before_decay 100000 --max_steps 200000 \\
    --save_freq 10000 --image_aug True
```

| 参数 | 默认值 | 含义 | 何时修改 |
|---|---|---|---|
| `--lam` | 0.1 | CAPRA 损失权重 | 成功率下降时减至 0.05 |
| `--beta` | 1.0 | 安全分布温度 | 通常不改，范围 0.5-2.0 |
| `--rho` | 0.5 | 前驱归因上权 | 设 0 关闭前驱上权 |
| `--capra_warmup_steps` | 500 | 预热步数 | 不稳定时增至 1000 |
| `--shuffle_buffer_size` | 2000 | **必须 ≤ 2000** | **绝对不要修改** |

添加 `--use_wandb True --wandb_entity 你的用户名` 启用 WandB 监控。正常训练时 `anchor_loss` 持续下降，`activation_ratio` 在 0.1-0.5 之间。

```bash
export CAPRA_CKPT=$(ls -td runs/*libero_spatial*capra*/step-* 2>/dev/null | head -1)
echo "CAPRA 检查点: $CAPRA_CKPT"
```

---

## 14. 实验 E：CAPRA 模型评估

**目的：** 评估 CAPRA 训练的模型，与实验 A 的 Baseline 对比。
**前提：** 实验 D 已完成（`$CAPRA_CKPT` 已设置）。

```bash
conda activate openvla && cd $PROJECT_ROOT

# 评估 CAPRA 模型
bash scripts/capra/eval_capra.sh "$CAPRA_CKPT" libero_spatial 50

# 对比 Baseline 与 CAPRA 结果
python -c "
import json, glob, os
for run_dir in sorted(glob.glob('experiments/logs/capra/CAPRA-EVAL-libero_spatial-*/'))[-2:]:
    f = run_dir + 'results_aggregate.json'
    if os.path.exists(f):
        d = json.load(open(f))
        name = run_dir.split('/')[-2][:50]
        print(name)
        print('  success_rate:', round(d.get('success_rate',0),3))
        print('  topple_rate: ', round(d.get('topple_rate',0),3))
        print('  support_break_rate:', round(d.get('support_break_rate',0),3))
"
```

**期望结果：**
- `success_rate`：CAPRA 与 Baseline 相近（差距 ≤ 3%）
- `topple_rate`：CAPRA 应显著低于 Baseline
- `support_break_rate`：CAPRA 应显著低于 Baseline

---

## 15. 实验 F：SafeLIBERO 评估

**目的：** 在 SafeLIBERO 基准上评估，该基准加入障碍物干预，专为安全性测试设计。

```bash
pip install safe-libero

# Baseline
python -m experiments.robot.capra.eval.run_capra_eval \\
    --pretrained_checkpoint tmp/models/openvla-7b-oft-finetuned-libero-spatial \\
    --task_suite_name libero_spatial --num_trials_per_task 50 \\
    --use_safe_libero True --use_l1_regression True \\
    --use_proprio True --num_images_in_input 2 \\
    --center_crop True --capra_eval_K 0 \\
    --run_id_note baseline_safelibero

# CAPRA 模型
python -m experiments.robot.capra.eval.run_capra_eval \\
    --pretrained_checkpoint "$CAPRA_CKPT" \\
    --task_suite_name libero_spatial --num_trials_per_task 50 \\
    --use_safe_libero True --capra_eval_K 0 \\
    --run_id_note capra_safelibero
```

---

## 16. 实验 G：程序化场景评估

**目的：** 用四种程序化模板系统测试特定副作用风险。

| 模板名称 | 测试什么 | 风险类型 |
|---|---|---|
| `collateral_clutter` | 抓取时是否避开附近易碰物体 | 附带碰撞 |
| `support_critical_neighbor` | 是否识别临界平衡物体 | 平衡临界 |
| `chain_reaction` | 是否预见多米诺连锁倒塌 | 连锁反应 |
| `occluded_remembered_hazard` | 是否记住视野外危险物 | 遮挡记忆 |

```bash
conda activate openvla && cd $PROJECT_ROOT

BASELINE="tmp/models/openvla-7b-oft-finetuned-libero-spatial"

for TEMPLATE in collateral_clutter support_critical_neighbor chain_reaction occluded_remembered_hazard; do
    echo "=== Baseline: $TEMPLATE ==="
    bash scripts/capra/eval_capra.sh "$BASELINE" libero_spatial 50 0 "$TEMPLATE"
    echo "=== CAPRA: $TEMPLATE ==="
    bash scripts/capra/eval_capra.sh "$CAPRA_CKPT" libero_spatial 50 0 "$TEMPLATE"
done
```

---

## 17. 实验 H：完整 CF 评估（SPIR/EAR）

**目的：** 获得 SPIR（安全偏好反转率）和 EAR（期望可避免风险）的真实非零值。
每步运行 K 次额外仿真，速度比 obs-only 慢约 K 倍。

### 检查 MuJoCo snapshot/restore 是否可用

```bash
conda activate openvla && cd $PROJECT_ROOT
python -c "
try:
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    from experiments.robot.libero.libero_utils import get_libero_env
    env, _ = get_libero_env(task, 'openvla', resolution=256)
    env.reset()
    state = env.sim.get_state()
    env.sim.set_state(state)
    print('MuJoCo snapshot/restore 可用，可跑完整 CF 评估')
except AttributeError as e:
    print(f'不可用: {e}，只能用 obs-only 模式（capra_eval_K=0）')
"
```

### CF 评估命令

```bash
# Baseline CF 评估（capra_eval_K=8，速度约慢 8 倍）
python -m experiments.robot.capra.eval.run_capra_eval \\
    --pretrained_checkpoint tmp/models/openvla-7b-oft-finetuned-libero-spatial 