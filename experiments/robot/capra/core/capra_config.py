"""
CAPRA 超参数配置中心 (capra_config.py)
=====================================

这是 CAPRA 系统的唯一配置入口。所有算法超参数、路径、训练参数
都在这里集中定义。通过 draccus CLI 或直接赋值来覆盖默认值。

设计原则
--------
- 所有默认值都是论文中的起始点，不是最优值。
- 在实际训练前，必须至少修改：vla_path、dataset_name、data_root_dir。
- shuffle_buffer_size 必须保持 <=2000，否则 TensorFlow RLDS 加载器
  会耗尽容器内存。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CAPRAConfig:
    # =========================================================
    # 路径配置
    # =========================================================
    # 挖掘缓存根目录：offline mining 产出的监督信号 JSON 存放位置
    cache_root: Path = Path("tmp/capra_cache")
    # 大型 rollout 输出（视频、轨迹）存放位置
    artifact_root: Path = Path("tmp/capra_artifacts")

    # =========================================================
    # 基础 VLA 模型引用
    # =========================================================
    # 用于 mining 的基线检查点路径（必须修改为实际路径）
    pretrained_checkpoint: str = "tmp/models/openvla-oft-libero"
    # 用于 finetune_capra.py 训练的起始检查点路径
    vla_path: str = "tmp/models/openvla-oft-libero"
    # RLDS 数据集名称，对应 tmp/datasets/rlds/{dataset_name}/
    dataset_name: str = "libero_spatial"
    # RLDS 数据集根目录
    data_root_dir: Path = Path("tmp/datasets/rlds")
    # 训练 run 输出根目录（checkpoints 会存到这里的子目录）
    run_root_dir: Path = Path("runs")

    # =========================================================
    # Rollout / Mining 参数
    # =========================================================
    # K：每个时间步采样的候选动作块数量
    # 越大越能找到更安全的等价动作，但 mining 时间线性增长
    K: int = 8

    # H_s：短时 counterfactual rollout 的执行步数
    # 用于评估每个候选动作的 P_t 和 F_t
    # 越大越准确，但 mining 速度越慢
    H_s: int = 5

    # 候选动作多样性噪声标准差
    # 在 nominal 动作块上叠加高斯噪声生成 K 个候选
    candidate_noise_sigma: float = 0.02

    # W：前驱归因的回看窗口步数
    # 用于在危险时间步前 W 步内寻找可替换的前驱动作
    W: int = 10

    # 前驱归因最大分析步数（控制计算预算）
    attribution_max_steps: int = 10
    # 每步最多尝试的替换候选数量
    attribution_max_replacements: int = 4
    # 每次替换 rollout 执行的步数 H_attr
    attribution_rollout_len: int = 8
    # 触发前驱归因的最小 F_t 阈值
    attribution_hazard_threshold: float = 0.10

    # 每个数据集划分挖掘的 episode 数量
    num_mining_episodes: int = 50

    # =========================================================
    # 任务等价集阈值
    # =========================================================
    # 绝对进度差阈值：|P_max - P_t(a)| <= epsilon_p_abs
    epsilon_p_abs: float = 0.05
    # 相对进度差阈值：|P_max - P_t(a)| / P_max <= epsilon_p_rel
    epsilon_p_rel: float = 0.10
    # 触发 CAPRA 损失的最低任务进度要求
    # 任务还没开始进行时不应该触发安全约束
    progress_floor: float = 0.20

    # =========================================================
    # 足迹 (Footprint) 分量权重
    # =========================================================
    # alpha_d：非目标物体位移权重（D_t 分量）
    alpha_d: float = 1.0
    # alpha_i：接触冲量权重（I_t 分量）
    alpha_i: float = 1.0
    # alpha_r：不可逆事件权重（R_t 分量）- 权重最高，因为不可逆
    alpha_r: float = 2.0

    # =========================================================
    # 训练损失参数
    # =========================================================
    # beta：安全目标分布的温度参数
    # beta 越大，q_hat 越集中在最安全的动作上（类似 softmin）
    beta: float = 1.0

    # rho：前驱归因上权因子
    # w_t = Delta_t * (1 + rho * R_t)，R_t 来自前驱归因
    # rho 越大，越鼓励模型在早期预防危险
    rho: float = 0.5

    # lambda (lam)：CAPRA 损失 vs Anchor 损失的权重比
    # L_total = L_anchor + lam * L_capra
    # 建议从 0.1 开始，稳定后可以尝试 0.3
    lam: float = 0.1

    # =========================================================
    # 训练超参数（从 FinetuneConfig 复制的相关字段）
    # =========================================================
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_steps: int = 200_000
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    save_freq: int = 10_000

    # ⚠️ 关键：必须保持 <= 2000
    # TensorFlow RLDS 数据加载器在 shuffle_buffer_size 过大时会耗尽内存
    # 导致容器崩溃，已在实践中验证
    shuffle_buffer_size: int = 2000

    image_aug: bool = True           # 是否启用图像增强
    use_l1_regression: bool = True   # 使用 L1 回归动作头（推荐）
    use_diffusion: bool = False      # 使用扩散动作头（实验性）
    use_film: bool = False           # 使用 FiLM 视觉条件化
    num_images_in_input: int = 2     # 输入图像数量（第三人称 + 手腕）
    use_proprio: bool = True         # 是否使用本体感知特征
    use_lora: bool = True            # 是否使用 LoRA 参数高效微调
    lora_rank: int = 32              # LoRA 秩
    lora_dropout: float = 0.0        # LoRA dropout
    center_crop: bool = True         # 是否中心裁剪
    num_open_loop_steps: int = 8     # 开环执行步数（动作块长度）
    unnorm_key: str = ""             # 动作反归一化键（空则自动检测）

    # =========================================================
    # 日志配置
    # =========================================================
    wandb_entity: str = "your-wandb-entity"   # 修改为你的 WandB 用户名
    wandb_project: str = "capra-openvla"       # WandB 项目名
    use_wandb: bool = False                    # 是否启用 WandB
    run_id_note: Optional[str] = None          # 运行备注（附加到 run 名称）
