"""
core.policy
============

A compact policy module that keeps only the components required by the new
training pipeline.  The default controller is a feed-forward MLP producing
world-frame accelerations, with an optional GRU layer for temporal context.

Design goals:
- single entry point `PolicyNetwork` with minimal configuration surface;
- pure-function helpers to initialise parameters and perform forward passes;
- explicit action bounding so the safety layer can reason about the same limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from core.flax_compat import struct


# ---------------------------------------------------------------------------
# Configuration and state containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dims: Tuple[int, ...] = (128, 128)
    activation: str = "relu"
    output_dim: int = 3  # acceleration in x/y/z
    use_rnn: bool = False
    rnn_hidden_size: int = 128
    action_limit: float = 5.0  # matches PhysicsParams.max_acceleration


@struct.dataclass
class PolicyState:
    hidden: jnp.ndarray


# ---------------------------------------------------------------------------
# Neural modules
# ---------------------------------------------------------------------------


def _activation(name: str):
    return {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "swish": nn.swish,
        "gelu": nn.gelu,
    }[name]


class FeedForwardPolicy(nn.Module):
    config: PolicyConfig

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = obs
        act = _activation(self.config.activation)
        for width in self.config.hidden_dims:
            x = nn.Dense(width, kernel_init=nn.initializers.xavier_uniform())(x)
            x = act(x)
        x = nn.Dense(self.config.output_dim, kernel_init=nn.initializers.xavier_uniform())(
            x
        )
        return jnp.tanh(x)


class RecurrentPolicy(nn.Module):
    config: PolicyConfig

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        hidden: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        act = _activation(self.config.activation)
        x = obs
        for width in self.config.hidden_dims[:-1]:
            x = nn.Dense(width, kernel_init=nn.initializers.xavier_uniform())(x)
            x = act(x)
        gru = nn.GRUCell(self.config.rnn_hidden_size)
        new_hidden, gru_out = gru(hidden, x)
        x = gru_out
        x = nn.Dense(self.config.hidden_dims[-1], kernel_init=nn.initializers.xavier_uniform())(
            x
        )
        x = act(x)
        x = nn.Dense(self.config.output_dim, kernel_init=nn.initializers.xavier_uniform())(
            x
        )
        return jnp.tanh(x), new_hidden


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    config: PolicyConfig

    def init_state(self, batch_size: int = 1) -> PolicyState:
        if not self.config.use_rnn:
            return PolicyState(hidden=jnp.zeros((batch_size, 0)))
        hidden = jnp.zeros((batch_size, self.config.rnn_hidden_size))
        return PolicyState(hidden=hidden)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        state: Optional[PolicyState] = None,
        training: bool = False,
    ):
        if self.config.use_rnn:
            assert state is not None, "RNN policy requires a PolicyState."
            rnn = RecurrentPolicy(self.config)
            action, new_hidden = rnn(obs, state.hidden, training=training)
            new_state = PolicyState(hidden=new_hidden)
        else:
            ff = FeedForwardPolicy(self.config)
            action = ff(obs, training=training)
            new_state = state

        # rescale tanh output to the physical action bounds
        action = action * self.config.action_limit
        return action, new_state


def initialise_policy_params(
    rng: jax.Array,
    config: PolicyConfig,
    observation_dim: int,
) -> Tuple[dict, PolicyState]:
    model = PolicyNetwork(config)
    dummy_obs = jnp.zeros((1, observation_dim))
    init_state = model.init_state(batch_size=1)
    params = model.init(rng, dummy_obs, init_state)
    return params, init_state


def policy_forward(
    params: dict,
    state: PolicyState,
    observations: jnp.ndarray,
    config: PolicyConfig,
    training: bool = False,
) -> Tuple[jnp.ndarray, PolicyState]:
    """
    Convenience wrapper to apply the policy outside of a Module context.
    """
    model = PolicyNetwork(config)
    action, new_state = model.apply(params, observations, state, training=training)
    return action, new_state


__all__ = [
    "PolicyConfig",
    "PolicyNetwork",
    "PolicyState",
    "initialise_policy_params",
    "policy_forward",
]
