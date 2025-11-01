#!/usr/bin/env bash
set -euo pipefail
# 简介：Kaggle GPU 训练入口脚本
# - 自动安装 JAX GPU 轮子（CUDA 12），再安装项目依赖
# - 之后将全部参数透传给 train_safe_policy.py
# 用法示例：
#   scripts/train_kaggle.sh --config-name stage1_efficiency --disable-safety \
#       --episodes 1500 --horizon 200 --stage-steps 500,500,500 \
#       --noise-levels 0.0,0.01,0.02 --point-cloud-modes ring,cylinder,box \
#       --target-distance-schedule 2.0,3.0,4.5 --target-angle-schedule 0,25,-25 \
#       --target-z-schedule 1.0,1.3,0.9 --success-eval-frequency 25 \
#       --output-dir outputs/efficiency_long_run

PY_BIN="python3"
PIP_BIN="pip"

# Kaggle 镜像通常预装 torch/TF；我们仅安装 JAX GPU 轮子与本项目依赖。
echo "[setup] installing JAX CUDA12 wheel..." >&2
${PIP_BIN} install -q "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# SciPy/OSQP 二进制兼容：优先使用 wheels，避免源码编译
# 若遇到二进制冲突，可适当放宽 requirements 中 scipy/osqp 的版本上限

echo "[setup] installing project requirements..." >&2
${PIP_BIN} install -q -r requirements.txt

# 可选：限制 XLA 显存占用比例，避免 OOM（按需开启）
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# 运行训练，参数透传
exec ${PY_BIN} -u train_safe_policy.py "$@"
