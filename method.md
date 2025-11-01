# 方法论总纲（修订版 v2）：基于可微安全层的敏捷飞行体系

本修订版在不改变总体方向的前提下，聚焦三件事：
1) 明确“梯度与职责”的边界：安全层（QP）仅向策略网络回传梯度，神经 CBF 通过专门的监督项学习；
2) 把“课程 + 守门 + 回退”固化为可执行的训练范式，解决零噪声成功率低与松弛依赖高；
3) 强化跨平台工程化（macOS/Kaggle GPU/Windows）与可复现性规范。

---

## 1. 愿景与研究目标（不变）
- 核心任务：构建单无人机在高度复杂、动态环境中执行任务时，兼顾绝对安全与高效路径的端到端学习系统。
- 理论基石：
  - GCBF+：以图结构控制障碍函数提供形式化安全证据；
  - Back to Newton’s Laws（DiffPhysDrone）：以可微分物理仿真 + BPTT 直接优化视觉-控制策略，实现高性能敏捷飞行。
- 应用场景：森林/仓储/城市峡谷等密集障碍，强调零通讯、低算力部署。

## 2. 设计原则（细化）
1. 模型一致性：安全层、策略与物理引擎使用一致动力学假设（点质量 + 控制盒约束 + 阻力/延迟可选），`dt` 与 `max_acc` 对齐；
2. 纯 JAX 实现：所有模块（感知、策略、QP、安全回退、渲染、物理）均为纯函数，`jit/vmap/grad/lax.scan` 友好；
3. 梯度边界清晰：
   - QP 安全层仅对“名义控制 u_nom”回传梯度，保证数值稳定与跨平台可复现；
   - 神经 CBF 的训练由“显式监督损失（safe/unsafe/ḣ/value-MSE）”承担；
   - 对 ∇h、H 在 rollout 中默认 `stop_gradient`，必要时作为实验开关；
4. 可微安全优先：安全层返回详尽诊断（违约、松弛、求解器状态、NaN 率），驱动课程自适应（加惩罚/降混合/早停回退）；
5. 课程式训练：效率→解析安全→神经 CBF→鲁棒扩展，逐阶段放开难度与噪声；
6. 指标守门：对成功率、最大违约、松弛指标、QP 退化率设硬阈值；超限即回退或终止，避免虚假“收敛”。

## 3. 系统总体架构（保持，补充梯度路径）
```
点云/深度 ─┬→ 图构建(GraphConfig) → GNN-CBF → h, ∇h, H ──────────┐
机体状态 ──┼────────────────────────────────────────┐   │
任务/目标 ──┼────────────────────→ 策略网络 (MLP/GRU) ─→ u_nom ─┼─→ QP 安全层 → u_safe → 物理引擎 → 下一状态
外部扰动 ──┘                                                         │                      │
                                                                    │                      └─ 轨迹/诊断/可视化
监督与损失： 效率损失 + 安全软约束 + CBF 监督（safe/unsafe/ḣ/value） → Optax 更新
梯度路径：  QP→仅对策略回梯度；CBF 通过监督损失获得梯度（推荐默认）。
```

## 4. 模块职责（增补要点）
- 感知/图构建（`core/perception.py`）：点云→图；GNN 输出标量 CBF，提供 h 的一阶/二阶导数；若 CBF 数值异常，则回退到 soft-min 解析 CBF；新增“cbf_nan_replaced_rate”诊断指标；
- 策略网络（`core/policy.py`）：MLP/GRU，输出世界系加速度，幅度受 `max_acc` 裁剪；支持教师蒸馏与速度对齐等正则；
- 可微安全层（`core/safety.py`）：解析主动集→OSQP→紧急制动 三段式求解；`custom_vjp` 仅把梯度回传至 u_nom；导出违约/松弛/失败/NaN/迭代数等；
- 物理引擎（`core/physics.py`）：点质量动力学，欧拉积分，时间梯度衰减（DiffPhys风格）；
- 训练循环（`core/loop.py`）：`lax.scan` 串联模块，可配置噪声、点云模式与混合权重（网络/解析 CBF）

## 5. 训练范式（四阶段 + 自适应守门）
1) 阶段A：效率基线（无安全层）
   - 目标：得到不依赖安全层的高质量名义策略（成功率≥0.98；末端速度低）。
   - 课程：多目标方向/高度/距离；点云模式以 ring 为主，适度增强（抖动/替换）。
   - 指标：成功率、最终距离、速度对齐、平滑度；保存 `policy_params.pkl`。

2) 阶段B：解析 CBF + QP
   - 目标：在解析 CBF 下稳定收敛，显著降低松弛使用率与违约峰值。
   - 策略：
     - 早期开启教师（PD 或蒸馏）以抑制“强投影+高松弛”的坏局部；
     - 松弛上限在前 40% episode 适度放宽，同时对“松弛均值”和“松弛激活次数”双惩罚；
     - 中期逐步收紧上限，后期固定为部署级别；
   - 自适应守门：
     - 定期评估 robust（多噪声），若 `neural_max_violation` 或训练期“NaN/Exceed/Fail”超阈，则自动：放大松弛与求解器惩罚（上限受限）、降低 CBF 混合权重（blend backoff）；
     - 触发硬阈值时回滚至最佳参数并终止当前阶段。

3) 阶段C：神经 CBF 联合（仅监督项回梯度）
   - 目标：在真实噪声/复杂点云下，神经 CBF 学到解析结构并提供更平滑、鲁棒的 h。
   - 路线：
     - Offline 预训练（soft-min 监督）→ Online 监督（safe/unsafe/ḣ/value-MSE）；
     - QP 仍仅对策略回梯度；∇h、H 继续 `stop_gradient`；
     - 混合权重从 0.5→0.8→1.0；
   - 守门：若“cbf_nan_replaced_rate”或“QP 失败率”升高，自动降混合/增惩罚/回滚。

4) 阶段D：鲁棒扩展与泛化
   - 目标：多噪声/遮挡/动态障碍/频率抖动等条件下保持高成功率；
   - 加入随机点云模式（cylinder/box/noise/mixed），目标距离课程，失真增强；
   - 评估多目标泛化（偏移目标、不同高度），辅以蒸馏或目标条件输入。

## 6. 课程与损失塑形（关键参数建议）
- 效率损失：目标到达（各向异性 XY/Z）、控制能耗、平滑度、悬停；
- 安全相关：
  - 软违约 `relu(-h)`；
  - 约束违约（QP residual）与松弛均值/次数分开计入；
  - CBF 监督：safe/unsafe 区间惩罚、ḣ + α h ≥ 0、value 对齐解析 CBF；
- 建议权重起点：
  - `safety_weight≈2.0`、`solver_violation_weight≈5.0`、`relaxation_weight≈1.0`、`relaxation_usage_weight≈0.5~2.0`；
  - `cbf_value_weight≈1.0`、`cbf_hdot_weight≈0.2`、`cbf_safe/unsafe≈0.5/0.5`、`cbf_weight_scale≈1.0`；
  - 逐阶段线性或分段调整，配合自适应守门。

## 7. 守门阈值与自适应策略（可执行）
- 在线指标（滚动窗口或每 N episode）：
  - `safety/qp_nan_rate ≤ 0.05`、`safety/relaxation_exceeded_rate ≤ 0.5`、`safety/qp_fail_rate ≤ 0.3`；
  - `safety/relaxation_mean ≤ 0.05`、`eval/success_rate ≥ 0.9`；
  - `robust/neural_max_violation ≤ 20`；
- 触发后动作：
  - 放大惩罚：`relax_penalty *= 1.5 (≤ 8.0)`、`solver_penalty *= 1.2 (≤ 5.0)`；
  - 降低混合：`blend = max(blend_min, blend - 0.2)`；
  - 若超硬阈值（或暖启动后仍超）：回滚至 best_params、记录原因并终止阶段。

## 8. 数据与点云（单机形态）
- 点云模式：`ring/cylinder/box/noise/mixed`；高度、半径、稠密度可控；
- 增强：抖动、随机替换、裁剪，保证合法高度与距离范围；
- 解析 CBF 审计：对随机状态统计解析 h 与真实最小距的相关性/偏保守量（温度、半径可校准）。

## 9. 评估与报告（对齐部署需求）
- 离线评估（每 N episode）：0/0.02/0.05 三档噪声，多次 rollout；统计成功率、最小 CBF、最大违约、松弛、能耗；
- 鲁棒评估：随机点云、遮挡、动态障碍、频率扰动；导出 `robust/*` 指标；
- 可视化：轨迹/控制/CBF/约束曲线、失败回放（GIF/视频）；
- 报告生成：`tools/collect_metrics.py + plot_metrics.py + report_generate.py` 自动汇总；`artifacts/` 保存 PDF/JSON 对比；
- 回gress检测：`tools/check_regression.py` 以 `success_rate ≥ 0.9`、`max_violation ≤ 20`、`relax_mean ≤ 0.05` 作为默认守门。

## 10. 跨平台与可复现（macOS/Kaggle/Windows）
- 版本与安装：
  - macOS CPU：按 `requirements.txt` 安装；
  - Kaggle GPU：先安装 JAX CUDA 轮子，再安装其余依赖（示例）
    - `pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
    - `pip install -r requirements.txt`
  - Windows：建议 `conda` 建环境（先装 `numpy/scipy`），再 `pip install jax`（CPU 版）与其余包；确保 `osqp` 安装成功；
- 一键脚本：提供 `scripts/train_cpu.sh`（本地小步验证）与 `scripts/train_kaggle.sh`（GPU 全量）；
- 可复现：
  - 固定随机种子、保存 `outputs/<run>/config.json` & 参数快照；
  - 保留 `conversation_log.txt` 与评估 JSON；
  - 严禁训练期 silent NaN：一旦检测到进入守门流程；
- 平台差异处理：对“数值微差导致的边缘失败”以“阈值缓冲+重试”处理，不以单次结果判死。

## 11. 实施里程碑（微调）
| 阶段 | 目标 | 指标与产出 |
|------|------|------------|
| M0  | 效率基线 | 成功率 ≥ 0.98、末端速度低，导出 `policy_params.pkl` |
| M1  | 解析安全 | 违约峰值 < 20、松弛均值 < 0.05，robust 指标过线，导出 `training_results.pkl` |
| M2  | 神经 CBF | blend→1.0 后性能不降，`cbf_nan_replaced_rate` 低、QP 失败率低 |
| M3  | 噪声鲁棒 | 多模式/多噪声成功率 ≥ 0.8，报告鲁棒性指标 |
| M4  | 工程交付 | 脚本/配置/模型打包，跨平台复现实验与技术报告 |

## 12. 风险与缓解（更新）
- CBF 学不出有效屏障：扩充预训练、加 value/ḣ/区间监督、Lipschitz 正则与裁剪；必要时降低混合权重回退到解析；
- QP 数值不稳：解析先行/OSQP 兜底/紧急制动，监控失败与残差，自动增惩罚与回退；
- 物理与实机不匹配：补充延迟/阻力/风扰建模，硬件在环标定参数；
- 训练振荡：学习率自适应、梯度裁剪、时间梯度衰减、课程回退；
- 数据/资产失控：配置快照、结果清理、回归守门；

## 13. 依赖与工具链（保持，补充建议）
- 核心库：`jax/jaxlib`、`flax`、`optax`、`jraph`、`ml_collections`、`omegaconf`、`numpy`、`scipy`、`osqp`；
- 建议：`chex`（测试）、`wandb`（追踪）、`tensorboard`（日志）、`trimesh/pyrender`（可选渲染）。

---

附：与实现的关键对齐
- QP 的自定义梯度：仅回传至策略（`core/safety.py`）；
- CBF 监督损失：在训练入口计算并合入总损失（`train_safe_policy.py`）；
- ∇h/H 的梯度控制：默认 `stop_gradient`（`core/loop.py`），可做成实验开关；
- 诊断指标：建议新增并记录 `cbf_nan_replaced_rate`，监控网络被解析回退替换的频度；
- 守门逻辑：自适应增惩罚/降混合/回滚与硬阈值早停应在训练循环中强制执行。
