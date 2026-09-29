"""Hyperparameter configuration for every agent.

Contents:
  - QLearningConfig : tabular Q-learning hyperparameters
  - DQNConfig       : DQN hyperparameters (Double DQN, Huber loss, schedules…)
  - PRESETS         : tuned per-environment overrides
  - make_config()   : defaults -> env preset -> user overrides, validated
  - parse_overrides(): turn CLI "key=value" strings into typed values

Hyperparameters live in frozen dataclasses instead of loose constructor
arguments so that (1) they are saved and restored as one unit, (2) typos in
overrides fail loudly, and (3) every run can be reported and reproduced.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QLearningConfig:
    """Tabular Q-learning."""

    n_bins: int = 10
    lr: float = 0.1
    gamma: float = 0.99
    # ε decays exponentially once per episode: ε ← max(ε_end, ε · decay).
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995

    def __post_init__(self) -> None:
        _check(self.n_bins >= 2, "n_bins must be >= 2")
        _check(0 < self.lr <= 1, "lr must be in (0, 1]")
        _common_checks(self.gamma, self.epsilon_start, self.epsilon_end)
        _check(0 < self.epsilon_decay <= 1, "epsilon_decay must be in (0, 1]")


@dataclass(frozen=True)
class DQNConfig:
    """Deep Q-Network with the standard stabilisers (Mnih et al., 2015;
    van Hasselt et al., 2016)."""

    lr: float = 5e-4
    gamma: float = 0.99
    hidden: tuple[int, ...] = (128, 128)
    batch_size: int = 128
    buffer_capacity: int = 100_000
    # Collect this many random-ish transitions before the first update.
    learning_starts: int = 1_000
    # One gradient step every `train_freq` environment steps.
    train_freq: int = 4
    # Hard-copy the online net into the target net every N *environment steps*.
    target_update_steps: int = 500
    # ε decays linearly per step from start to end over `epsilon_decay_steps`.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    double_dqn: bool = True
    huber_loss: bool = True
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        _check(self.lr > 0, "lr must be > 0")
        _common_checks(self.gamma, self.epsilon_start, self.epsilon_end)
        _check(
            len(self.hidden) >= 1 and all(h > 0 for h in self.hidden),
            "hidden must be a non-empty tuple of positive ints",
        )
        _check(self.batch_size > 0, "batch_size must be > 0")
        _check(self.buffer_capacity >= self.batch_size, "buffer_capacity must be >= batch_size")
        _check(self.learning_starts >= 0, "learning_starts must be >= 0")
        _check(self.train_freq >= 1, "train_freq must be >= 1")
        _check(self.target_update_steps >= 1, "target_update_steps must be >= 1")
        _check(self.epsilon_decay_steps >= 1, "epsilon_decay_steps must be >= 1")
        _check(self.max_grad_norm > 0, "max_grad_norm must be > 0")


AgentConfig = QLearningConfig | DQNConfig

CONFIG_CLASSES: dict[str, type[QLearningConfig] | type[DQNConfig]] = {
    "qlearning": QLearningConfig,
    "dqn": DQNConfig,
}

# Tuned starting points. Anything not listed falls back to the class default.
PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("qlearning", "CartPole-v1"): {"n_bins": 10, "lr": 0.1, "epsilon_decay": 0.998},
    ("qlearning", "LunarLander-v3"): {"n_bins": 8, "epsilon_decay": 0.9995},
    ("dqn", "CartPole-v1"): {
        "lr": 1e-3,
        "hidden": (128, 128),
        "learning_starts": 1_000,
        "train_freq": 1,
        "target_update_steps": 500,
        "epsilon_decay_steps": 10_000,
    },
    ("dqn", "LunarLander-v3"): {
        "lr": 5e-4,
        "hidden": (256, 256),
        "batch_size": 128,
        "learning_starts": 5_000,
        "train_freq": 4,
        "target_update_steps": 1_000,
        "epsilon_decay_steps": 100_000,
        "epsilon_end": 0.05,
    },
}


def make_config(
    agent_type: str, env_id: str, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """Class defaults, then the env preset, then explicit overrides."""
    cls = CONFIG_CLASSES[agent_type]
    values = {**PRESETS.get((agent_type, env_id), {}), **(overrides or {})}
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(
            f"Unknown hyperparameter(s) for {agent_type}: {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(valid))}"
        )
    return cls(**values)


def parse_overrides(agent_type: str, pairs: list[str]) -> dict[str, Any]:
    """Parse ["lr=0.001", "hidden=64,64"] using each field's declared type."""
    cls = CONFIG_CLASSES[agent_type]
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or key not in types:
            raise ValueError(f"Bad override {pair!r}. Use key=value with key in {sorted(types)}")
        out[key] = _coerce(str(types[key]), raw.strip())
    return out


def to_dict(cfg: AgentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def from_dict(agent_type: str, data: dict[str, Any]) -> AgentConfig:
    cls = CONFIG_CLASSES[agent_type]
    data = dict(data)
    if "hidden" in data:
        data["hidden"] = tuple(data["hidden"])
    return cls(**data)


# ── helpers ──────────────────────────────────────────────────────────────


def _coerce(type_name: str, raw: str) -> Any:
    if "tuple" in type_name:
        return tuple(int(x) for x in raw.split(",") if x.strip())
    if "bool" in type_name:
        if raw.lower() in {"1", "true", "yes", "si", "sí"}:
            return True
        if raw.lower() in {"0", "false", "no"}:
            return False
        raise ValueError(f"Not a boolean: {raw!r}")
    if "int" in type_name:
        return int(float(raw))  # accepts "1e5"
    if "float" in type_name:
        return float(raw)
    return raw


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _common_checks(gamma: float, eps_start: float, eps_end: float) -> None:
    _check(0 <= gamma <= 1, "gamma must be in [0, 1]")
    _check(0 <= eps_end <= eps_start <= 1, "need 0 <= epsilon_end <= epsilon_start <= 1")
