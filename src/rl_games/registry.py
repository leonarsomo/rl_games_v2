"""Agent lookup, save paths, and construction.

Contents:
  - SAVE_DIR        : directory holding saved agents (env var RLGAMES_SAVE_DIR)
  - AGENTS          : agent key -> ("module:Class", save-file extension)
  - AGENT_CHOICES   : the valid agent keys
  - save_path()     : save file for an (agent, env) pair
  - log_path()      : CSV training log next to the save
  - agent_class()   : import and return a registered agent class
  - create()        : build a fresh, untrained agent from a config
  - load()          : load a saved agent
  - load_or_create(): load if a save exists, else create

Agent classes are named as strings and imported on demand, so commands that
never touch the DQN do not import torch.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rl_games.config import make_config

if TYPE_CHECKING:
    from rl_games.agents.base import BaseAgent

SAVE_DIR = Path(os.environ.get("RLGAMES_SAVE_DIR", "saves"))

# agent key -> ("module:Class", save-file extension). Adding an agent = a row.
AGENTS: dict[str, tuple[str, str]] = {
    "qlearning": ("rl_games.agents.qlearning:QLearningAgent", ".npz"),
    "dqn": ("rl_games.agents.dqn:DQNAgent", ".pt"),
}

AGENT_CHOICES = tuple(AGENTS)


def _slug(env_id: str) -> str:
    return env_id.replace("/", "_")


def save_path(agent_type: str, env_id: str, save_dir: Path | None = None) -> Path:
    """One save file per (agent, env) pair."""
    suffix = AGENTS[agent_type][1]
    return (save_dir or SAVE_DIR) / f"{agent_type}_{_slug(env_id)}{suffix}"


def best_path(agent_type: str, env_id: str, save_dir: Path | None = None) -> Path:
    """Best greedy checkpoint written by `train --eval-every`."""
    p = save_path(agent_type, env_id, save_dir)
    return p.with_name(p.stem + "_best" + p.suffix)


def log_path(agent_type: str, env_id: str, save_dir: Path | None = None) -> Path:
    return (save_dir or SAVE_DIR) / f"{agent_type}_{_slug(env_id)}.csv"


def agent_class(agent_type: str) -> type[BaseAgent]:
    """Import and return the class registered under `agent_type`."""
    module_name, _, class_name = AGENTS[agent_type][0].partition(":")
    return getattr(import_module(module_name), class_name)


def create(
    agent_type: str,
    env_id: str,
    overrides: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> BaseAgent:
    """Build a fresh, untrained agent (defaults → env preset → overrides)."""
    cfg = make_config(agent_type, env_id, overrides)
    return agent_class(agent_type)(env_id, cfg, seed=seed)


def load(agent_type: str, env_id: str, save_dir: Path | None = None) -> BaseAgent:
    """Load the saved agent for this (agent, env) pair."""
    return agent_class(agent_type).load(save_path(agent_type, env_id, save_dir))


def load_or_create(
    agent_type: str,
    env_id: str,
    overrides: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
    save_dir: Path | None = None,
) -> BaseAgent:
    """Resume from a save if there is one, otherwise start fresh.

    Overrides only apply to a fresh agent: silently changing the
    hyperparameters of a half-trained agent would make its log misleading.
    """
    if save_path(agent_type, env_id, save_dir).exists():
        if overrides:
            raise ValueError(
                "A save already exists; hyperparameter overrides would not apply. "
                f"Run 'rlgames delete {agent_type} --env {env_id}' first."
            )
        return load(agent_type, env_id, save_dir)
    return create(agent_type, env_id, overrides, seed=seed)
