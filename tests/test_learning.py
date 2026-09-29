"""End-to-end: do the agents actually learn? (slow, a few seconds each)."""

from __future__ import annotations

import numpy as np
import pytest

from rl_games import evaluate, registry

pytestmark = pytest.mark.slow


def test_qlearning_learns_cartpole() -> None:
    agent = registry.create("qlearning", "CartPole-v1", seed=0)
    agent.train(1500, verbose=False)
    score = np.mean(evaluate.run_episodes(agent, n_episodes=10, seed=1000))
    assert score > 60  # random policy scores ~22


def test_dqn_learns_cartpole() -> None:
    agent = registry.create("dqn", "CartPole-v1", seed=0)
    agent.train(220, verbose=False)
    score = np.mean(evaluate.run_episodes(agent, n_episodes=10, seed=1000))
    assert score > 100
