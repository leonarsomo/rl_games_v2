"""Agents. Import lazily via rl_games.registry so torch loads only when needed."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_games.agents.dqn import DQNAgent
    from rl_games.agents.qlearning import QLearningAgent

__all__ = ["DQNAgent", "QLearningAgent"]


def __getattr__(name: str) -> object:
    if name == "DQNAgent":
        from rl_games.agents.dqn import DQNAgent

        return DQNAgent
    if name == "QLearningAgent":
        from rl_games.agents.qlearning import QLearningAgent

        return QLearningAgent
    raise AttributeError(name)
