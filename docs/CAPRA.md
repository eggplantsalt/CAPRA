这个文件是用于让code agent时刻掌握全局信息，也就是知道我们最后想做成一件什么样的工程项目

你们当前已经锁死的关键点包括：研究对象是 manipulation-only VLA 的内生安全；核心对象是 local avoidable-risk regret；主方法是 CAPRA；主指标是 SPIR 和 EAR；主评测用 SafeLIBERO，底座任务来自 LIBERO，再补四个轻量程序化 side-effect 模板；同时明确不做 test-time shield、不做 memory 主模块、不做 generic preference/DPO 主叙事、不做 mechanistic steering 主贡献。

另外，OpenVLA-OFT 当前公开 repo 的组织方式很适合“旁挂式扩展”：LIBERO evaluation 在 `experiments/robot/libero/`，通用 robot helper 在 `experiments/robot/`，训练入口在 `vla-scripts/finetune.py`，README quick start 直接从 `experiments.robot.libero.run_libero_eval`、`experiments.robot.openvla_utils` 和 `prismatic.vla.constants` 导入组件。所以 CAPRA 最稳的接法不是大改上游，而是新增 `experiments/robot/capra/` 和一个 CAPRA 专用训练/评测入口，再只在必要处对 `prismatic` 做小钩子修改。([GitHub][1])

SafeLIBERO 本身也已经足够当主安全 benchmark 用，它把四个 LIBERO suite 扩成 16 个任务、32 个场景、1600 个 episode，并区分 Level I/II 的障碍干预强度；这正适合你们证明“不开 test-time safety layer，也能降低 intrinsic avoidable risk”。AEGIS/VLSA 那条线是 plug-and-play 的 safety constraint layer，不是你们的方法目标，只把它当 benchmark 来源和对比对象。([VLSA][2])

下面我直接给你可用版本。

---

## 一、给 code agent 的全局上下文（直接复制进 `context.md`）

# CAPRA on OpenVLA-OFT: Global Context

## 1. 项目身份

这个项目不是从零开始想 idea，而是在一个已经基本冻结的研究方案上做工程实现。我们要在 OpenVLA-OFT 上实现 CAPRA，用于 manipulation-only VLA 的 intrinsic safety。我们研究的问题不是“模型有没有看到危险物体”，也不是“怎样加一个外部安全层拦住危险动作”，而是：

当若干候选动作对任务推进几乎等价时，模型是否仍系统性地把更危险的动作排在更前面？

我们把这种现象形式化为 local avoidable-risk regret。我们要证明并利用这样一个事实：unsafe manipulation 往往不是最后一步偶然出事，而是少数几个早期局部排序错误沿时间传播形成 precursor chain。

## 2. 明确不做什么

不要把这个项目做成以下任何一种东西：

第一，不要做外接 safety shield、CBF、QP、安全过滤层、test-time verifier、point-cloud obstacle detector 之类的外挂式安全模块。AEGIS/VLSA 是对比线，不是实现目标。

第二，不要把主创新做成 explicit memory module、object memory、long-horizon memory。Safety Alternative Buffer 只允许作为 training-time mining tool，不得进入 test-time 模型结构。

第三，不要把主叙事写成 generic preference alignment、pairwise ranking、DPO for safety。CAPRA 的核心是 constrained local risk projection，不是“偏好学习”表面故事。

第四，不要把主贡献做成 mechanistic steering、unsafe neurons、某些 layer/head 的推理时干预。可解释性只允许作为 analysis/appendix 支线。

第五，不要走 CMDP/SafeRL 主线，不要把训练写成“又一个 SafeVLA/ISA 式的外层安全 RL 优化”。

## 3. 选定的模型、基准和代码底座

基础模型选 OpenVLA-OFT。基础 manipulation 任务族选 LIBERO。主安全评测选 SafeLIBERO。我们自己的新增不是大 benchmark，而是四个 reset-time 轻量程序化 side-effect 模板：

Collateral Clutter
Support-Critical Neighbor
Chain Reaction
Occluded Remembered Hazard

可选的 robustness sanity check 可以留接口给 LIBERO-X，但它不是第一优先级。

OpenVLA-OFT 公开 repo 当前的关键入口是：

`experiments/robot/libero/run_libero_eval.py`
`experiments/robot/libero/libero_utils.py`
`experiments/robot/openvla_utils.py`
`experiments/robot/robot_utils.py`
`vla-scripts/finetune.py`

repo 根目录已经有 `experiments/robot`、`prismatic`、`vla-scripts` 等主干目录，所以 CAPRA 应该作为 overlay 接入，而不是推翻上游目录结构。([GitHub][1])

## 4. 对 OpenVLA-OFT 的实现假设

OpenVLA-OFT 的默认 LIBERO 路径是 action-chunk continuous policy：README quick start 显式通过 MLP action head 生成 continuous actions；官方 LIBERO fine-tuning 命令默认 `--use_l1_regression True`、`--use_diffusion False`。因此，CAPRA 的训练实现不要假设“有解析 token logits 的离散动作分布”，而应默认走 candidate-set approximation / sample-set distillation 版本。([GitHub][1])

这意味着在某个状态 t，我们用有限候选集近似局部分布。若拿不到精确 base density，可以先实现 uniform prior 版本，但接口要允许未来替换成 sample score 或 likelihood proxy。这个选择在科学上是允许的，不改变 CAPRA 主体。

## 5. CAPRA 的冻结 pipeline

### 5.1 rollout 数据来源

先让 base VLA 在仿真环境里自然执行，收集 rollout。每条 rollout 至少保存：

视觉观测
语言指令
历史动作
环境状态快照
对象位姿变化
接触/冲量
支撑关系或可近似 support predicates
topple/fall/workspace violation 等信号

这些全部来自 simulator，不依赖人工安全标注。

### 5.2 候选动作集

在时刻 t，构造候选动作集 A\_t。候选来源有两部分：

一部分来自当前 policy 自己的 sample-K action chunks。
另一部分来自 Safety Alternative Buffer。

Safety Alternative Buffer 只允许作为 training-time 的离线候选挖掘器。测试时它绝不进入模型结构。其检索表征可以用冻结的 pre-action visual-language embedding 与低维几何摘要拼接。

### 5.3 短反事实回放

对每个候选动作 a，从同一状态快照出发，强制执行这个动作块，再向前滚一个短视界 H\_s。对每个候选，计算：

`P_t(a)`：task progress
`F_t(a)`：footprint

注意，`P_t(a)` 不是简单 raw reward，而是 progress potential 的变化；要用 simulator/task 自身的阶段信号或可解释的进度势函数来实现，不要偷换成泛化奖励。

### 5.4 footprint 的三部分

`F_t(a)` 必须由三个 manipulation-specific 成分组成：

1. non-target displacement
2. contact impulse
3. irreversible events

并且 footprint 设计要尽量 object-wise、decomposable、task-conditioned。只对当前阶段“不该动”的对象收费，不要把所有环境变化混成一个拍脑袋的 scalar。

### 5.5 任务等效动作集

不是所有更安全动作都值得学。只有当几个动作在任务推进上确实差不多时，才比较它们的安全性。因此必须构造 task-equivalent set `E_t`。

实现时采用双尺度规则：
既看绝对进度差，也看相对进度差；
同时要求最佳候选进度至少达到一个 progress floor。

如果 `P_max` 太低，或者 `E_t` 为空，这个状态不触发 CAPRA loss。

### 5.6 局部可避免风险对象

在 `E_t` 内找 footprint 最小的动作，得到局部最安全任务等效动作。基于它定义 `Delta_t`，即该时刻模型多承担了多少本可避免的局部风险。我们最终评估的主对象是 local avoidable-risk regret，而不是简单 collision count。

### 5.7 安全目标分布

不要把主训练目标写成 pairwise preference。最终目标是安全目标分布，也就是局部受约束风险投影。实现时可写为：

`q_hat_t(a_i) ∝ prior(a_i) * exp(-beta * F_t(a_i)) * 1[a_i in E_t]`

在连续动作场景里，这个分布定义在有限候选集上，而不是假装有一个解析连续分布。

### 5.8 precursor attribution

除了局部 `Delta_t`，还要实现 delayed hazard 的前因归责。做法是：对危险轨迹在一个窗口 W 内回看，逐个把候选高风险步替换成更安全替代动作，做 budgeted counterfactual rollout，看哪个替换能最大幅度降低后续危险。目标不是找唯一 culprit，而是得到可重复的 precursor chain 和责任分数 `R_t`。

### 5.9 最终训练目标

最终训练必须采用 dense anchor + sparse CAPRA activation：

所有状态都保留 anchor / trust-region to base policy。
只有当 `P_max` 达到 floor、`E_t` 非空、且 `Delta_t > 0` 时，CAPRA 项才激活。

权重写成：

`w_t = Delta_t * (1 + rho * R_t)`

总损失写成：

`L = L_anchor + lambda * sum_t w_t * KL(q_hat_t || pi_theta(.|h_t,l))`

测试时模型结构不能增加任何新安全层。

## 6. 指标冻结

正文主指标只有两个：

SPIR：Safety Preference Inversion Rate
EAR / J\_AR：Expected Avoidable Risk

机制验证指标有两个：

AttributionEditGain
PrecursorLeadTime

外部结果指标包括：

success
protected-object displacement
topple
support-break

不要在正文里无节制堆指标。SPIR + EAR 是主打，其余用于机制与外部一致性。

## 7. 工程拆解

工程上至少要实现以下模块：

1. simulator snapshot save/restore
2. env/state signal API
3. candidate action generation
4. short-horizon counterfactual rollout
5. task progress computation
6. footprint computation
7. task-equivalence filtering
8. Safety Alternative Buffer
9. budgeted long-horizon precursor attribution
10. offline CAPRA cache / dataset builder
11. CAPRA training integration
12. CAPRA evaluation metrics

最大工程量在 counterfactual rollout infrastructure。一旦它搭起来，监督基本都能自动生成。

## 8. 推荐的代码集成原则

请优先采用“新目录 + 小钩子”的最小侵入策略：

在 `experiments/robot/capra/` 放所有 CAPRA 专属逻辑。
在 `vla-scripts/` 下增加 `finetune_capra.py`。
在 `experiments/robot/capra/` 下增加 `run_capra_mining.py` 和 `run_capra_eval.py`。
在 `tests/capra/` 放 CAPRA 的单元测试和 smoke test。
大体量 rollout/cache 产物不要直接放进 git，路径应可配置。
对 `prismatic/` 的修改必须保持最小，只在训练 hook 或 loss 注入不可避免时才动。

基线 `run_libero_eval.py` 和 `finetune.py` 最好保持可单独运行，不要为了 CAPRA 破坏原 baseline 路径。这个项目要的是“在现有 OpenVLA-OFT 上新增 CAPRA”，不是“重写 OpenVLA-OFT”。这是工程原则，不是论文贡献。([GitHub][1])

## 9. 实现时允许的近似与边界

如果 simulator 没有直接暴露某些信号，比如精确支撑关系或精确冲量，可以做最小可解释近似，但必须：

明确区分 exact signal 和 approximated signal；
把近似封装在单独模块里；
不要把近似写死在训练主循环里；
在进度文件里记录近似假设。

同样，如果 exact snapshot/restore 难以做到，也应先尝试底层 state round-trip；只有确实不可能时，才退到“可重复重建同一场景”的最小替代方案，并显式记录保真度限制。

## 10. 初版实现的交付标准

一个阶段性可接受的 CAPRA 实现，至少要满足：

baseline LIBERO eval 仍可原样运行；
CAPRA mining 脚本能从若干 rollout 里挖出非空 supervision；
训练脚本能在一个小 batch 上同时跑出 anchor loss 和 CAPRA loss；
评测脚本能输出 SPIR、EAR、EditGain、LeadTime、success、protected-object displacement、topple、support-break；
整个系统不引入 test-time shield，不引入 inference-time memory，不依赖人工 safety label。

## 11. 超参数地位

以下量仍然可以作为工程超参数调，但它们不是换方向的理由：

K
H\_s
W
epsilon\_p
alpha\_d / alpha\_i / alpha\_r
beta / rho / lambda
progress floor
buffer 检索表征
precursor attribution 的预算化策略

请把它们做成 config，不要把它们硬编码到方法定义里。

## 12. code agent 的工作风格要求

先读代码，再改代码。不要猜未打开文件。
默认实施更改，而不是只给建议。
每次只完成当前阶段，不要擅自跨阶段继续。
保持最小侵入，避免过度工程化。
不要删除或重写无关代码。
每完成一个阶段，更新 `progress_capra.md` 和 `capra_state.json`。
每次输出必须包含：读过哪些文件、改了哪些文件、跑了什么验证、还剩什么 blocker。
任何破坏性或难撤销的操作都要先停下来问。

analysis-only 支线（如 future-footprint probe、relation diagnostics）不是初版实现优先级。它们最多是后续附录分析，不应该阻塞 CAPRA 主 pipeline。

---

## 二、推荐的项目目录方案

这个目录方案的核心逻辑是：沿着 OpenVLA-OFT 现有的 `experiments/robot` + `vla-scripts` 组织方式，做 CAPRA 的旁挂扩展，而不是把 baseline 入口揉烂。公开 repo 现在把 LIBERO eval 和通用 robot helper 分开管理，训练也有独立入口，所以你给 code agent 的目录要求最好和这个风格对齐。([GitHub][1])

```text
openvla-oft/
├─ experiments/
│  └─ robot/
│     ├─ libero/
│     │  ├─ run_libero_eval.py              # baseline，尽量不破坏
│     │  └─ libero_utils.py                 # baseline，尽量只复用
│     ├─ openvla_utils.py                   # baseline helper
│     ├─ robot_utils.py                     # baseline helper
│     └─ capra/
│        ├─ __init__.py
│        ├─ config.py
│        ├─ env_adapter.py                  # 统一包一层 LIBERO / SafeLIBERO env
│        ├─ snapshot.py                     # save/restore state
│        ├─ state_api.py                    # 读取对象、位姿、接触、support 等信号
│        ├─ object_roles.py                 # target / protected / non-target / irrelevant
│        ├─ task_progress.py                # P_t(a)
│        ├─ signals.py                      # displacement / impulse / irreversible events
│        ├─ footprint.py                    # F_t(a)
│        ├─ equivalence.py                  # task-equivalent set E_t
│        ├─ candidate_actions.py            # sample-K actions + policy prior interface
│        ├─ buffer.py                       # Safety Alternative Buffer（训练期 only）
│        ├─ rollout.py                      # short CF rollout / long replacement rollout
│        ├─ precursor.py                    # precursor attribution, R_t, chain
│        ├─ mining_cache.py                 # 离线 cache schema / io
│        ├─ build_capra_dataset.py          # 将 mining 结果转训练样本
│        ├─ metrics.py                      # SPIR / EAR / EditGain / LeadTime / external metrics
│        ├─ procedural_splits.py            # 四类 side-effect 模板
│        ├─ run_capra_mining.py             # 离线 supervision 挖掘入口
│        ├─ run_capra_eval.py               # CAPRA 评测入口
│        └─ report_utils.py
├─ vla-scripts/
│  ├─ finetune.py                           # baseline
│  └─ finetune_capra.py                     # CAPRA 训练入口，优先新建而不是重写 baseline
├─ tests/
│  └─ capra/
│     ├─ test_snapshot.py
│     ├─ test_state_api.py
│     ├─ test_footprint.py
│     ├─ test_equivalence.py
│     ├─ test_candidate_actions.py
│     ├─ test_precursor.py
│     ├─ test_metrics.py
│     └─ test_smoke_pipeline.py
├─ scripts/
│  └─ capra/
│     ├─ mine_capra.sh
│     ├─ train_capra.sh
│     ├─ eval_capra.sh
│     └─ smoke_capra.sh
├─ progress_capra.md
└─ capra_state.json
```

我自己的建议是：大缓存、大 rollout 产物、大 supervision 文件不要塞进 repo 内部目录，统一走 `--cache_root` / `--artifact_root` 配置，避免 git 污染。这个是工程建议。

---

## 三、每轮都带的“通用执行前言”

这个前言不是必须，但非常有用。你每次给 code agent 发阶段 prompt 时，都把这段放在最前面。

```text
你现在是在一个已有的 openvla-oft 代码库里工作。默认直接实施更改，不要只给建议。

在回答任何关于代码库结构、入口、类名、函数名、训练流程的判断之前，先读取相关文件；不要猜未打开的代码。若我提到的路径不存在，请你自己定位等价文件，然后继续，不要停在泛泛建议上。

工作原则：
1. 最小侵入。优先新增 CAPRA 相关文件，尽量不破坏 baseline 的 LIBERO eval 和原始 finetune 路径。
2. 只做当前阶段要求的事，不要擅自跨到后续阶段。
3. 不要过度工程化。不要为假设中的未来需求加多余抽象。
4. 不要写占位性空实现。要么实现，要么在非常窄的边界上显式抛出 NotImplementedError，并写清原因。
5. 所有临时脚本、实验性文件如果不再需要，请在任务结束前清理。
6. 任何破坏性或难撤销操作（删除文件、大规模重构、改 baseline 行为）都先停下来问我。

状态跟踪要求：
- 创建并持续更新 `progress_capra.md`
- 创建并持续更新 `capra_state.json`
- 每完成一次编辑后，简短汇报：
  a) 读了哪些文件
  b) 改了哪些文件
  c) 跑了哪些测试或 smoke commands
  d) 现在还剩哪些 blocker / 风险

验证要求：
- 优先写最小但真实的单元测试和 smoke test
- 不要为了让测试通过而硬编码逻辑
- 如果底层环境或依赖不完整，至少让纯 Python 单测与接口层 smoke test 跑通，并清楚说明哪一步受外部依赖限制

输出风格要求：
- 先做事，再汇报
- 汇报要简洁、事实化
- 不要写很长的空泛总结
```

---
