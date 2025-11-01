#!/usr/bin/env bash
set -euo pipefail
# 简介：本地 CPU/MPS 小步验证入口脚本
# 用法示例：
#   scripts/train_cpu.sh --config-name stage1_efficiency --disable-safety --episodes 20 --horizon 200 --output-dir outputs/efficiency_cpu_h200
# 该脚本不会写死任何路径，所有训练参数透传给 train_safe_policy.py。

# 选择 Python 解释器：优先使用 .venv，其次系统 python3
PY_BIN="$(
  if [[ -x ".venv/bin/python" ]]; then echo ".venv/bin/python"; 
  elif command -v python3 >/dev/null 2>&1; then echo "python3"; 
  else echo "python"; fi
)"

echo "[info] using python: ${PY_BIN}" >&2
exec "${PY_BIN}" -u train_safe_policy.py "$@"
