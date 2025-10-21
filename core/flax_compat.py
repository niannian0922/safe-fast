"""
为旧版 JAX 提供兼容补丁，以确保 Flax 正常运行。

较新的 Flax 期望 `jax.config.define_bool_state`，而在较新的 JAX 中该函数被
重命名为 `register_bool_state`。在 JAX 0.4.x 上该符号可能不存在，因此在导入
Flax 之前先做一个简易垫片。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


if not hasattr(jax.config, "define_bool_state"):
    if hasattr(jax.config, "register_bool_state"):
        def _define_bool_state(name: str, default: bool, helpstring: str):
            jax.config.register_bool_state(name, default, helpstring)
            if not hasattr(jax.config, name):
                setattr(jax.config, name, default)
            return getattr(jax.config, name)
        jax.config.define_bool_state = _define_bool_state  # type: ignore[attr-defined]
    else:
        def _define_bool_state(name: str, default: bool, helpstring: str):
            if not hasattr(jax.config, name):
                setattr(jax.config, name, default)
            return getattr(jax.config, name)
        jax.config.define_bool_state = _define_bool_state  # type: ignore[attr-defined]

# 确保 `jax.linear_util` 可用（Flax 会导入它）
if not hasattr(jax, "linear_util"):
    try:
        from jax._src import linear_util as _linear_util  # type: ignore
    except ImportError:  # pragma: no cover
        _linear_util = None
    if _linear_util is not None:
        jax.linear_util = _linear_util  # type: ignore[attr-defined]

# 为旧版 JAX 提供 Flax 期望的 KeyArray 别名
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jnp.ndarray  # type: ignore[attr-defined]

if not hasattr(jax.random, "default_prng_impl"):
    try:
        from jax._src import prng as _prng  # type: ignore
    except ImportError:  # pragma: no cover
        pass
    else:
        default_impl = None
        if hasattr(_prng, "default_prng_impl"):
            default_impl = _prng.default_prng_impl  # type: ignore[attr-defined]
        else:
            default_impl = getattr(_prng, "threefry_prng_impl", None)
        if default_impl is not None:
            jax.random.default_prng_impl = lambda: default_impl  # type: ignore[attr-defined]

from flax import struct  # noqa: E402  (在补丁后再导入)

__all__ = ["struct"]
