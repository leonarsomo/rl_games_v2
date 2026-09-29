"""Running an agent without learning from it.

Contents:
  - run_episodes() : play N episodes greedily, returning each episode's return
  - summarize()    : mean / std / min / max of a list of returns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from rl_games import envs

if TYPE_CHECKING:
    from rl_games.agents.base import BaseAgent


def run_episodes(
    agent: BaseAgent,
    env: gym.Env | None = None,
    n_episodes: int = 10,
    *,
    seed: int | None = None,
) -> list[float]:
    """Run `n_episodes` greedily (ε ignored) and return each episode's total reward.

    If `env` is None a fresh one is created for `agent.env_id` and closed
    afterwards. Pass `seed` to evaluate on a fixed, reproducible set of
    starting states (episode i uses seed + i).
    """
    own_env = env is None
    env = env or envs.make(agent.env_id)
    returns: list[float] = []
    try:
        for i in range(n_episodes):
            obs, _ = env.reset(seed=None if seed is None else seed + i)
            done, total = False, 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += float(reward)
            returns.append(total)
    finally:
        if own_env:
            env.close()
    return returns


def summarize(returns: list[float]) -> dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
