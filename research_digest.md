# Safe Agile Flight Internal Research Digest

## Objective Snapshot
- **Goal**: Fuse GCBF+-style formal safety guarantees with differentiable-physics policy learning to produce a single-drone controller that maintains strict safety while achieving aggressive flight through cluttered environments.
- **Core Workflow**: Perception graph encoder (GNN) → policy head (Flax MLP/RNN) → differentiable safety layer (QP solved by `qpax`) → JAX-native differentiable physics engine → BPTT training calibrated by multi-objective losses and robust gradient handling.
- **Non-negotiables**: End-to-end JAX implementation, BPTT via `jax.lax.scan`, qpax-based safety filtering with failover, MGDA-style loss balancing, time-gradient decay, curriculum and adversarial crash correction.

## Architectural Blueprint (from `method.md`)
- **Perception (core/perception.py)**: Import GCBF+ GNN, refactor to single-agent LiDAR graph via JAX KNN/radius search; outputs CBF value `h(x)` and gradients.
- **Policy (core/policy.py)**: Flax MLP/RNN producing nominal action `u_nom`; supports recurrent state for history-sensitive flight.
- **Safety Layer (core/safety.py)**:
  - Builds QP with CBF constraint (Actuation via qpax).
  - Implements three-tier failover: qpax solution → conservative backup → emergency stop + policy penalty.
  - Ensures differentiability + JIT compatibility.
- **Physics Engine (core/physics.py)**: JAX-native point-mass dynamics with thrust delay, drag, actuator smoothing; exposures `next_state = f(state, action)` signature.
- **Training Loop (core/loop.py)**: `jax.lax.scan` rollouts with checkpointing & gradient clipping; integrates time-gradient decay (exponential factor) and MGDA weights.
- **Losses**:
  - Physics-driven (velocity tracking, obstacle clearance, control smoothness) from DiffPhysDrone.
  - CBF-compliance penalties plus violation slack tracking.
  - Additional curriculum, noise scheduling, ACC pipeline controlled via `main.py`.
- **Tooling**: qpax for differentiable QP, Optax optimizers, Flax modules, JAX random control for data augmentation.

## Paper: *Back to Newton's Laws* (Key Insights)
- **Differentiable Simulation**: Point-mass quadrotor model with thrust lag, drag; depth rendering integrated into the computational graph; gradients backpropagate through full simulation.
- **Training Regimen**:
  - 64 parallel randomized environments, 150 steps, Δt ≈ 1/15 s.
  - Physics-driven loss = velocity tracking (Smooth L1 on averaged speed error), obstacle avoidance (softplus + quadratic barrier weighted by approach speed), control smoothness (acceleration & jerk terms).
  - Temporal gradient decay combats explosion; implemented as exponential factor during reverse pass.
- **Policy**: CNN encoder over downsampled depth, fused with kinematic state, recurrent GRU; outputs thrust commands with predicted velocity head for odom-free operation.
- **Results**: Zero-shot sim2real; 90% success in dense scenes, 20 m/s outdoor flights, multi-agent emergent coordination without communication.
- **Adaptation Targets**:
  - Reproduce differentiable physics & rendering path in JAX.
  - Preserve loss composition and gradient decay within `jax.lax.scan` backward pass.
  - Maintain parallel environment randomization & curriculum schedules.

## Code: `DiffPhysDrone` Repository Highlights
- **`env_cuda.py`**: Custom CUDA autograd functions (`RunFunction`) for dynamics; gradient decay wrapper `GDecay`; randomized obstacle generation; LiDAR-like depth rendering; support for multi-drone batches, variable speed caps, wind, actuator bias.
- **`main_cuda.py`**: Training loop orchestrating environment reset, depth rendering, policy forward pass, QP-free safety heuristics, loss accumulation, gradient step via AdamW + Cosine LR; handles video logging.
- **`model.py`**: Lightweight Conv stack + GRU; integrates kinematic state through linear projection; small output head scaling.
- **Takeaways for JAX port**:
  - Need JAX analog of `quadsim_cuda` for differentiable physics.
  - Depth processing uses `3 / clamp(depth) - 0.6` normalization with noise + pooling (stride 4) → preserve for fairness.
  - Actuator buffer / lag and random control frequency jitter essential for robustness.

## Paper: *GCBF+* (Key Insights)
- **Graph Control Barrier Functions (GCBF)**: Defines safety over local subgraph; attention weights enforce locality (zero influence beyond sensing radius) enabling scalability and arbitrary neighbor counts.
- **Safety Guarantee**: Single learned GCBF certifies safety for any MAS size; proof handles neighbor-set discontinuities.
- **Training Framework**:
  - Jointly learn CBF `h_θ` and distributed policy `π_ϕ` via GNNs.
  - Loss components: derivative condition (Eq. 6), safe set positivity, unsafe set negativity, control imitation (toward QP solution), hinge margins `γ`.
  - Uses look-ahead horizon to ensure actuation feasibility (labels safe/unsafe based on future rollout under action limits).
- **QP Role**: Offline solves CBF-QP using learned `h_θ` to produce target safe control `u_QP`; ensures compatibility between safety and goal-reaching.
- **Observations Handling**: LiDAR rays as graph nodes; goal nodes integrated for target conditioning; message passing via attention ensures permutation invariance.

## Code: `gcbfplus` Repository Highlights
- **Framework**: Entirely JAX/Flax + jraph + optax; modular separation of environments, GNNs, algorithms, training utilities.
- **`gcbfplus/nn`**: Attention-based message passing; `GNNLayer` composes message MLP, attention gate, node update MLP; reuse for both CBF and policy heads.
- **`algo/gcbf_plus.py`**:
  - Wraps CBF (Flax module) and deterministic policy; maintains train & target params; uses `MaskedReplayBuffer` with unsafe prioritization.
  - Computes qp targets via `JaxProxQP` (requires gradients); constructs QP with relaxation variables for feasibility.
  - Loss terms mirror paper: safe/unsafe hinge, derivative via forward rollout, action imitation, statistics for monitoring.
  - `safe_mask` infers horizon-valid safe states; `unsafe_mask` derived from env; training uses inner epochs and chunked batches for memory control.
- **`env`**: Multiple multi-agent dynamics (Single/Double Integrator, Dubins, LinearDrone, CrazyFlie); provide graph building, affine dynamics, LiDAR rays, reference controls.
- **Utilities**: Graph handling (`GraphsTuple`), masking, jit-friendly helpers; crucial when transplanting modules.
- **Lessons for our system**:
  - Reuse Flax GNN stack for perception `h, ∇h` (single-agent adaptation by constructing ego-centric graph from LiDAR/KNN).
  - Replace `JaxProxQP` with `qpax` while preserving API (requires mapping matrices H, g, G, h to qpax format, plus differentiable fallback path).
  - Replay buffer & horizon-safe labeling guide curriculum and unsafe prioritization.

## Integration Strategy & Action Items
- **Perception Rebuild**
  - Port GCBF+ GNN modules into `core/perception.py`, adapt node/edge features for LiDAR point cloud + ego state.
  - Implement JAX KNN/radius search to build fixed-size local graphs; ensure attention weights go to zero near sensing radius for Definition 1 compliance.
- **Safety Layer**
  - Translate QP formulation: decision `[u, slack]`, objective `||u - u_nom||^2 + ρ||slack||_1`, constraints `∂h/∂x (f+gu) + α h >= -slack`.
  - Wrap qpax solve in pure JAX function; add auto fallback when solver fails (emergency braking + penalty term in loss).
- **Physics Engine**
  - Reimplement point-mass dynamics with thrust filtering, actuator lag, drag, random wind; maintain differentiability.
  - Integrate depth rendering surrogate (e.g., voxelized obstacles → differentiable SDF depth) compatible with JIT.
- **Training Loop**
  - Construct `scan_function(carry, inputs)` returning `(carry, traced_outputs)`; include time-gradient decay using custom `lax.custom_vjp` or multiply partials inside backward.
  - Implement MGDA weighting between physics loss and CBF loss; integrate gradient clipping and multi-phase curriculum (noise schedule, adversarial crash correction).
  - Ensure on-policy data feeds safe/unsafe buffers similar to GCBF+ for stabilizing CBF learning.
- **Losses & Monitoring**
  - Physics: velocity tracking, obstacle proximity, acceleration, jerk; replicate DiffPhysDrone formulas.
  - Safety: derivative slack, hinge penalties, qpax failure counters.
  - Logging: success metrics, constraint violation counts, qpax relaxation magnitudes, gradient norms.
- **Tooling & Infrastructure**
  - Adopt Optax optimizers, Flax Linen modules, JAX random seeding per rollout.
  - Create comprehensive config (3-stage curriculum, noise injection, evaluation hooks).
  - Build evaluation scripts for both sim rollouts and safety margin statistics.

## Risk & Open Questions
- **qpax Stability**: Need numeric guardrails (warm-start, regularization, scaling of constraint Jacobians). Consider autodiff-friendly damping and solver timeout detection.
- **Depth Rendering in JAX**: Identify approach (e.g., JAX-based ray marching, precomputed SDF grids) that remains differentiable and performant.
- **Gradient Decay Implementation**: Evaluate custom VJP vs. explicit rescaling of Jacobians to avoid interfering with JAX AD.
- **Perception Graph Construction**: Single-agent LiDAR → dynamic neighbor sets; must ensure smooth attention transitions to avoid gradient spikes.
- **Data Labeling for Safe/Unsafe**: Need JAX-compatible rollout lookahead under action limits to classify states (mirroring GCBF+ horizon logic).
- **Real-to-Sim Gap**: Incorporate stochastic wind, sensor noise, actuator bias consistent with DiffPhysDrone logs; maintain logging to calibrate sim realism.

## Immediate Next Steps
1. Scaffold new JAX project layout (`core/`, `configs/`, `tools/`) matching method blueprint.
2. Port GCBF+ GNN modules into Flax within repository, create single-drone graph builder utilities.
3. Prototype qpax-based safety layer on simplified dynamics to validate gradients & fallback logic.
4. Implement point-mass physics + differentiable depth pipeline, verify against DiffPhysDrone metrics.
5. Assemble `lax.scan` training loop with placeholder losses, then incrementally integrate physics + safety objectives.
6. Establish monitoring dashboards (TensorBoard/Array logging) to track constraint adherence and optimization health.

