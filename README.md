# Safe Agile Flight – 当前研发状态与执行指南

本仓库是单无人机安全敏捷飞行项目的研究平台，实现了以下关键组件：

- **JAX 可微分物理引擎**（`core/physics.py`）
- **Flax 策略网络**（`core/policy.py`）
- **GNN CBF 模块 + qpax 安全层**（`core/perception.py`, `core/safety.py`）
- **统一训练入口**（`train_safe_policy.py`），支持 Optax 训练、解析/神经 CBF、噪声课程
- **辅助工具**：预训练解析 CBF（`tools/pretrain_cbf.py`）、结果汇总（`tools/stage_summary.py`）

以下内容帮助你快速了解目前的代码结构、已完成的工作、如何复现实验，以及下一步改进方向。

---

## 目录结构与职责

- `core/`
  - `physics.py`: JAX 原生点质量动力学、时间梯度衰减。
  - `policy.py`: Flax MLP/GRU 策略。
  - `perception.py`: 构建点云图、GNN CBF；解析 CBF 回退。
  - `safety.py`: 带松弛变量的 CBF-QP（qpax），失败时优雅回退。
  - `loop.py`: JAX `lax.scan` BPTT 循环，串联策略→安全→物理。
  - `simple_training.py`: 效率损失项（目标、能耗、平滑、悬停）。
- `configs/default_config.py`: 默认训练设置（噪声、损失权重、课程）。
- `train_safe_policy.py`: 主训练脚本，支持 CLI 参数控制噪声/课程、安全开启、CBF 预训练、输出目录等。
- `tools/pretrain_cbf.py`: 使用解析 CBF 对神经 CBF 网络进行监督预训练（目前为批处理版本，可多次运行累积参数）。
- `tools/stage_summary.py`: 汇总训练输出目录（Pickle）的指标（loss、约束违背等），支持 `--json` 输出。
- `outputs/`: 默认训练产出目录（例如 `efficiency_test/`, `safety_cbf_pretrained/` 等）。
- `PROJECT_STATUS_DETAIL.md`: 最新的阶段性总结、限制、未来计划。
- `conversation_log.txt`: 研发过程日志。

---

## 环境准备

1. **Python 版本**：建议 3.10（已在 macOS CPU 上验证）。  
2. **虚拟环境**（推荐）：`python -m venv .venv`，然后 `source .venv/bin/activate`（Windows:`.venv\Scripts\activate`）。
3. **安装依赖**：`pip install -r requirements.txt`。  
   - CPU 训练：默认 `jax`, `jaxlib` 为 CPU 版。
   - GPU 训练：请参考 JAX 官方说明，安装对应 CUDA 版本的 `jax`/`jaxlib`。
4. **（如需复现特定版本）**：可使用 `pip freeze > requirements.lock` 或提供 `poetry/conda` 配置。

---

## 现有成果概览

| 输出目录 | 说明 | 关键指标 |
|----------|------|-----------|
| `outputs/efficiency_test/` | 纯效率训练（安全关闭） | loss ≈ 1.70e3，BPTT 链路正常，未统计成功率（仅 50 步示例）。 |
| `outputs/safety_cbf_pretrained/` | 解析 CBF + 无噪声安全 | loss ≈ 2.55e3，约束违背 ≈ 1e-9。 |
| `outputs/safety_noise_cbf/` | 预训 CBF + 三阶段噪声课程 | loss ≈ 3.16e3，约束违背 ≈ 0.28（尚需调参）。 |
| `outputs/cbf_pretrained.pkl` | 最新 CBF 预训练权重（解析监督） | loss 约在 0.09–0.23 范围，供训练加载。 |

使用 `python tools/stage_summary.py <目录> --json` 可得到上述指标的 JSON 摘要。

---

## 使用说明

### 1. 效率训练（纯策略）

```bash
python train_safe_policy.py \
  --episodes 1000 \
  --horizon 80 \
  --safety-weight 0.0 --solver-weight 0.0 --relaxation-weight 0.0 \
  --disable-curriculum \
  --noise-levels 0.0 \
  --stage-steps 1000 \
  --disable-safety \
  --output-dir outputs/efficiency_long_run
```

> 提示：当前示例只有 50 步，未达到高成功率；建议按上述方式延长训练步数与 horizon，并增加成功率统计。

### 2. 解析 CBF 安全训练

```bash
python train_safe_policy.py \
  --episodes 300 \
  --horizon 60 \
  --safety-weight 0.05 --solver-weight 0.02 --relaxation-weight 0.01 \
  --stage-steps 300 \
  --noise-levels 0.0 \
  --output-dir outputs/safety_stage
```

### 3. 噪声课程训练（解析或预训 CBF）

```bash
python train_safe_policy.py \
  --episodes 300 \
  --horizon 60 \
  --safety-weight 0.05 --solver-weight 0.02 --relaxation-weight 0.01 \
  --stage-steps 100,100,100 \
  --noise-levels 0.0,0.02,0.05 \
  --cbf-params outputs/cbf_pretrained.pkl \  # 可选：加载预训练 CBF
  --output-dir outputs/safety_noise
```

### 4. CBF 预训练（解析监督）

```bash
# 从头开始
python tools/pretrain_cbf.py --dataset-size 1024 --steps 2000 --batch-size 64 \
  --output outputs/cbf_pretrained.pkl --log-every 200

# 继续训练
python tools/pretrain_cbf.py --dataset-size 1024 --steps 2000 --batch-size 64 \
  --init-params outputs/cbf_pretrained.pkl \
  --output outputs/cbf_pretrained.pkl --log-every 200
```

### 5. 结果汇总

```bash
python tools/stage_summary.py outputs/safety_noise --json
```

### 6. 安全评估与自适应调节

- 训练过程中会自动根据课程阶段生成成功率噪声与违约阈值；一旦超阈，框架会放大松弛/违约惩罚并降低神经 CBF 与解析 CBF 的混合比例，相关指标记录在 `adaptive/*`。
- 新增鲁棒性扫描：可通过 `--robust-eval-frequency`、`--robust-eval-noise`、`--robust-eval-trials` 控制噪声评估频率与强度；日志中的 `robust/*` 指标分别给出当前混合策略与纯神经策略下的最大违约值。
- 若需定制惩罚力度，可使用 `--blend-backoff`、`--blend-min`、`--relax-boost`、`--solver-boost`、`--relax-alert` 等参数；全部阈值都会写入 `eval/violation_threshold`，便于与历史结果对照。

---

## 环境与跨平台注意事项

- 当前训练主要在 macOS CPU（MacBook Air）上验证。  
- 若需要在 Windows/Linux 上运行，请确保：
  1. 使用 `requirements.txt` 或 freeze 文件保持一致性；
  2. 安装合适的 JAX/JAXLIB（CPU 或 GPU 版本）；
  3. 使用相同随机种子（`config.seed` 等）以减少浮点差异；
  4. 如果在 GPU 上运行，建议升级到匹配 CUDA 版本的 `jax`/`jaxlib`；
  5. 可将保存的训练成果（策略参数、CBF 预训练参数）上传至 GitHub 或共享盘，方便他人加载。
- 如果团队成员在性能更强的服务器/GPU 上训练，可直接加载你提供的成果参数；Optax 默认每次重新初始化优化器状态，若要“续训”，可在未来扩展脚本保存 `opt_state`。

---

## 当前限制与下一步计划

### 现存限制
1. **效率训练成功率低**：目前 50 步示例未显示明确成功率（最终距离 ≈ 1.5m）；需延长训练步数和调整损失权重后再进入安全阶段。
2. **神经 CBF 在噪声下的收敛不充分**：噪声课程下约束违背仍高（0.2~0.3），说明预训练或安全权重还需优化。
3. **预训练脚本仍为循序批处理**：虽支持大数据量，但耗时较长；未来要改写成向量化/JIT。
4. **自动化流程缺失**：效率→安全→噪声一键执行尚未实现，需要进一步脚本化。

### 下一步（短期）
1. **长程效率训练**：延长 episodes/horizon、调节损失权重，统计成功率，达到“几乎完美”再进入安全阶段。
2. **优化 CBF 预训练**：将 `tools/pretrain_cbf.py` 向量化/JIT，实现更大规模训练；继续记录解析 loss。
3. **噪声权重扫描**：编写批量脚本测试不同 `stage_steps`、`noise_levels`、`safety_weight` 组合，寻找“低违背 + 可接受效率”的权衡点。

### 中期
1. **自动化脚本**：将效率→安全→噪声训练封装成统一指令，便于 CI。  
2. **CBF 预训练强化**：在解析监督基础上尝试边界样本、梯度监督等，提升 GNN 表达。
3. **跨平台验证**：提供 Windows/Linux/GPU 部署指南及测试脚本，确保团队成员能无缝训练。

### 长期
1. **更大规模实验**：换 GPU/集群进行长程训练，结合噪声、课程、联合优化。  
2. **完整神经 CBF**：当预训练稳定后，尝试完全依赖神经 CBF，评估解析回退的必要性。  
3. **自动化报告**：构建训练结果的自动化分析（曲线、指标表）和可视化工具。

---

## 常见命令速查

| 目的 | 命令示例 |
|------|-----------|
| 纯效率训练 | `python train_safe_policy.py --episodes 1000 --disable-safety ...` |
| 解析 CBF 安全 | `python train_safe_policy.py --episodes 300 ...` |
| 载入预训 CBF + 噪声 | `python train_safe_policy.py --cbf-params outputs/cbf_pretrained.pkl ...` |
| CBF 预训练 | `python tools/pretrain_cbf.py --dataset-size 1024 --steps 2000 ...` |
| 结果汇总 | `python tools/stage_summary.py outputs/safety_noise --json` |
| 冒烟测试 | `pytest -q tests/test_smoke.py` |

> 更多细节请参考 `PROJECT_STATUS_DETAIL.md` 或运行 `python train_safe_policy.py --help`、`python tools/pretrain_cbf.py --help`。

---

## 联系 / 贡献

- 欢迎在 GitHub 提交 Issue 或 Pull Request，或在团队内部按需求讨论。  
- 如需进一步帮助（例如脚本自动化、跨平台部署、训练调参），请联系当前维护者。  

谢谢使用，祝研究顺利！  
