"""
Helper to keep Flax compatible with older JAX releases.

Recent Flax versions expect `jax.config.define_bool_state`, which was renamed
to `register_bool_state` in newer JAX builds.  On JAX 0.4.x the symbol may be
absent, so we provide a small shim before importing Flax.
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

# Ensure `jax.linear_util` is available (Flax imports it)
if not hasattr(jax, "linear_util"):
    try:
        from jax._src import linear_util as _linear_util  # type: ignore
    except ImportError:  # pragma: no cover
        _linear_util = None
    if _linear_util is not None:
        jax.linear_util = _linear_util  # type: ignore[attr-defined]

# Provide KeyArray alias expected by Flax when using older JAX releases
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

from flax import struct  # noqa: E402  (import after patching)

__all__ = ["struct"]
