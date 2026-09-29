from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rl_games import registry
from rl_games.agents.dqn import DQNAgent, QNetwork
from rl_games.agents.replay import ReplayBuffer
from rl_games.config import DQNConfig

SMALL = DQNConfig(
    hidden=(16, 16),
    batch_size=8,
    buffer_capacity=500,
    learning_starts=16,
    train_freq=1,
    target_update_steps=50,
    epsilon_decay_steps=200,
)


def test_qnetwork_shapes() -> None:
    net = QNetwork(4, 2, (32, 16))
    out = net(torch.zeros(5, 4))
    assert out.shape == (5, 2)
    n_linear = sum(isinstance(m, torch.nn.Linear) for m in net.net)
    assert n_linear == 3


def test_replay_buffer_wraps_and_samples() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=4, obs_dim=2, rng=rng)
    for i in range(6):
        buf.push(np.full(2, i), i % 2, float(i), np.full(2, i + 1), i == 5)
    assert len(buf) == 4
    b = buf.sample(3)
    assert b.states.shape == (3, 2)
    assert b.states.dtype == np.float32
    assert set(b.rewards.tolist()) <= {2.0, 3.0, 4.0, 5.0}  # oldest two overwritten
    with pytest.raises(ValueError, match="Cannot sample"):
        buf.sample(10)


def test_learn_step_reduces_loss_on_fixed_batch() -> None:
    agent = DQNAgent("CartPole-v1", SMALL, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(64):
        s = rng.normal(size=4).astype(np.float32)
        agent.buffer.push(s, int(rng.integers(2)), 1.0, s, True)  # target is exactly 1
    losses = [agent._learn() for _ in range(200)]
    assert losses[-1] < losses[0] * 0.2


def test_terminal_mask_removes_bootstrap() -> None:
    cfg = DQNConfig(
        hidden=(8,), batch_size=1, buffer_capacity=1, learning_starts=0, huber_loss=False
    )
    agent = DQNAgent("CartPole-v1", cfg, seed=0)
    s = np.zeros(4, dtype=np.float32)
    with torch.no_grad():  # make the target net predict a huge value
        for p in agent.target_net.parameters():
            p.fill_(0.0)
        agent.target_net.net[-1].bias.fill_(1000.0)
    agent.buffer.push(s, 0, 1.0, s, True)
    q_before = float(agent.q_values(s)[0])
    loss = agent._learn()
    # With terminated=True the target is r = 1, not 1 + 0.99 * 1000.
    assert loss == pytest.approx((q_before - 1.0) ** 2, rel=1e-4)


def test_epsilon_linear_schedule_and_target_sync() -> None:
    agent = DQNAgent("CartPole-v1", SMALL, seed=0)
    agent.total_steps = 100
    agent._on_step_end()
    assert agent.epsilon == pytest.approx(1.0 + 0.5 * (0.05 - 1.0))
    agent.total_steps = 10_000
    agent._on_step_end()
    assert agent.epsilon == pytest.approx(0.05)


def test_save_load_roundtrip_with_custom_hidden() -> None:
    agent = DQNAgent("CartPole-v1", SMALL, seed=0)
    agent.train(3, verbose=False)
    path = registry.save_path("dqn", "CartPole-v1")
    agent.save(path)
    loaded = DQNAgent.load(path)
    assert loaded.config.hidden == (16, 16)  # the original project lost this
    assert loaded.training_episodes == 3
    s = np.ones(4, dtype=np.float32)
    np.testing.assert_allclose(loaded.q_values(s), agent.q_values(s), rtol=1e-6)


def test_export_json_forward_matches_torch() -> None:
    agent = DQNAgent("CartPole-v1", SMALL, seed=0)
    spec = json.loads(json.dumps(agent.export_json()))
    x = np.array([0.1, -0.2, 0.03, 0.4])
    h = x
    for i, layer in enumerate(spec["layers"]):
        h = np.asarray(layer["W"]) @ h + np.asarray(layer["b"])
        if i < len(spec["layers"]) - 1:
            h = np.maximum(h, 0)
    np.testing.assert_allclose(h, agent.q_values(x.astype(np.float32)), atol=1e-4)
