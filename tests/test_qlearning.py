from __future__ import annotations

import numpy as np
import pytest

from rl_games import registry
from rl_games.agents.qlearning import QLearningAgent
from rl_games.config import QLearningConfig


@pytest.fixture
def agent() -> QLearningAgent:
    return QLearningAgent("CartPole-v1", QLearningConfig(n_bins=10), seed=0)


def test_discretize_is_hashable_and_in_range(agent: QLearningAgent) -> None:
    key = agent.discretize(np.array([0.0, 0.0, 0.0, 0.0]))
    assert isinstance(key, tuple)
    assert all(isinstance(k, int) for k in key)
    assert all(0 <= k < 10 for k in key)


def test_discretize_clips_out_of_range(agent: QLearningAgent) -> None:
    low = agent.discretize(np.array([-99.0, -99.0, -99.0, -99.0]))
    high = agent.discretize(np.array([99.0, 99.0, 99.0, 99.0]))
    assert low == (0, 0, 0, 0)
    assert high == (9, 9, 9, 9)


def test_lunarlander_binary_flags_pass_through() -> None:
    a = QLearningAgent("LunarLander-v3", QLearningConfig(n_bins=6), seed=0)
    key = a.discretize(np.array([0, 0.5, 0, 0, 0, 0, 1.0, 0.0]))
    assert len(key) == 8
    assert key[-2:] == (1, 0)


def test_td_update_matches_formula(agent: QLearningAgent) -> None:
    s, s2 = (1, 1, 1, 1), (2, 2, 2, 2)
    agent.q_table[s2][:] = [1.0, 3.0]
    agent._observe(s, 0, reward=1.0, next_state=s2, terminated=False)
    expected = 0.0 + 0.1 * (1.0 + 0.99 * 3.0 - 0.0)
    assert agent.q_table[s][0] == pytest.approx(expected)


def test_terminal_transition_does_not_bootstrap(agent: QLearningAgent) -> None:
    s, s2 = (1, 1, 1, 1), (2, 2, 2, 2)
    agent.q_table[s2][:] = [100.0, 100.0]
    agent._observe(s, 1, reward=1.0, next_state=s2, terminated=True)
    assert agent.q_table[s][1] == pytest.approx(0.1 * 1.0)


def test_greedy_and_exploration(agent: QLearningAgent) -> None:
    s = (0, 0, 0, 0)
    agent.q_table[s][:] = [0.0, 5.0]
    assert agent.select_action(s, deterministic=True) == 1
    agent.epsilon = 1.0
    actions = {agent.select_action(s) for _ in range(200)}
    assert actions == {0, 1}


def test_epsilon_decays_per_episode_with_floor(agent: QLearningAgent) -> None:
    agent.epsilon = 0.010001
    agent._on_episode_end()
    assert agent.epsilon == pytest.approx(0.01)


def test_save_load_roundtrip(agent: QLearningAgent) -> None:
    agent.train(5, verbose=False)
    path = registry.save_path("qlearning", "CartPole-v1")
    agent.save(path)
    loaded = QLearningAgent.load(path)
    assert loaded.training_episodes == 5
    assert loaded.total_steps == agent.total_steps
    assert loaded.epsilon == pytest.approx(agent.epsilon)
    assert loaded.config == agent.config
    assert set(loaded.q_table) == set(agent.q_table)
    for k, v in agent.q_table.items():
        np.testing.assert_allclose(loaded.q_table[k], v)


def test_seeded_training_is_reproducible() -> None:
    a = QLearningAgent("CartPole-v1", seed=123)
    b = QLearningAgent("CartPole-v1", seed=123)
    assert a.train(20, verbose=False).rewards == b.train(20, verbose=False).rewards
