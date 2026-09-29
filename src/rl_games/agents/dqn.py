"""Deep Q-Network (DQN) in PyTorch, with the standard stabilisers.

Contents:
  - QNetwork : MLP mapping a state to one Q-value per action
  - DQNAgent : ε-greedy policy (linear per-step schedule), replay buffer,
               Double-DQN target, Huber loss, gradient clipping, step-based
               target-network sync, and safe (weights_only) save/load.

One gradient step, with Double DQN (van Hasselt et al., 2016):

    a*  = argmax_a' Q_online(s', a')                  # online net selects
    y   = r + γ · Q_target(s', a*) · (1 − terminated)  # target net evaluates
    L   = Huber(Q_online(s, a), y)

Plain DQN (Mnih et al., 2015) uses max_a' Q_target(s', a') instead, which
systematically over-estimates Q-values; set `double_dqn=False` to compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import nn

from rl_games.agents.base import BaseAgent
from rl_games.agents.replay import ReplayBuffer
from rl_games.config import DQNConfig, from_dict


class QNetwork(nn.Module):
    """Fully-connected state → Q(s, ·). No activation on the output: Q is unbounded."""

    def __init__(self, state_dim: int, action_dim: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        width = state_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch of states, shape (batch, action_dim)."""
        return self.net(x)


def pick_device() -> torch.device:
    """CUDA if present; otherwise CPU. (For nets this small, MPS is slower.)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(BaseAgent[np.ndarray, DQNConfig]):
    agent_type = "dqn"
    label = "DQN"

    def __init__(
        self, env_id: str, config: DQNConfig | None = None, *, seed: int | None = None
    ) -> None:
        super().__init__(env_id, config or DQNConfig(), seed=seed)
        cfg = self.config
        self.device = pick_device()
        if seed is not None:
            torch.manual_seed(seed)

        self.q_net = QNetwork(self.state_dim, self.n_actions, cfg.hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.n_actions, cfg.hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # never trained directly

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.loss_fn: nn.Module = nn.SmoothL1Loss() if cfg.huber_loss else nn.MSELoss()
        self.buffer = ReplayBuffer(cfg.buffer_capacity, self.state_dim, self.rng)

    def _setup(self, env: Any) -> None:
        shape = env.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError(f"DQN expects a flat vector observation, got shape {shape}")
        self.state_dim = int(shape[0])

    # ── policy ─────────────────────────────────────────────────────────

    def _to_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Q(s, ·) for a single state -- handy for inspection and the web export."""
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q_net(x).squeeze(0).cpu().numpy()

    def _greedy_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.q_values(state)))

    # ── learning ───────────────────────────────────────────────────────

    def _observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> float | None:
        self.buffer.push(state, action, reward, next_state, terminated)
        cfg = self.config
        ready = len(self.buffer) >= max(cfg.learning_starts, cfg.batch_size)
        if ready and self.total_steps % cfg.train_freq == 0:
            return self._learn()
        return None

    def _learn(self) -> float:
        """Sample a mini-batch and perform one gradient step. Returns the loss."""
        cfg = self.config
        b = self.buffer.sample(cfg.batch_size)
        dev = self.device
        states = torch.as_tensor(b.states, device=dev)
        actions = torch.as_tensor(b.actions, device=dev).unsqueeze(1)
        rewards = torch.as_tensor(b.rewards, device=dev)
        next_states = torch.as_tensor(b.next_states, device=dev)
        terminated = torch.as_tensor(b.terminated, device=dev)

        q_sa = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if cfg.double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + cfg.gamma * next_q * (1.0 - terminated)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _on_step_end(self) -> None:
        cfg = self.config
        # Linear ε schedule over environment steps (as in the DQN paper).
        frac = min(1.0, self.total_steps / cfg.epsilon_decay_steps)
        self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
        if self.total_steps % cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ── persistence ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Weights, optimizer and metadata. The replay buffer is not saved,
        so a resumed run re-fills it (after `learning_starts` steps)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._common_state()
        data["config"]["hidden"] = list(self.config.hidden)
        data.update(
            q_net_state=self.q_net.state_dict(),
            target_net_state=self.target_net.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            state_dim=self.state_dim,
            action_dim=self.n_actions,
        )
        torch.save(data, path)
        print(f"Saved {self.label} agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        # weights_only=True refuses to unpickle arbitrary objects, so a
        # downloaded .pt file cannot execute code on load.
        data = torch.load(path, weights_only=True, map_location="cpu")
        cls._check_format(data, path)
        cfg = from_dict(cls.agent_type, data["config"])
        assert isinstance(cfg, DQNConfig)
        agent = cls(data["env_id"], cfg, seed=data.get("seed"))
        agent._restore_common(data)
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        return agent

    def _info_extra(self) -> list[str]:
        params = sum(p.numel() for p in self.q_net.parameters())
        return [
            f"  Network params   : {params:,}",
            f"  Device           : {self.device}",
            f"  Replay buffer    : {len(self.buffer)}/{self.config.buffer_capacity}",
        ]

    def _progress_extra(self) -> str:
        return f" | Buffer: {len(self.buffer)}"

    # ── export for the HTML5 simulator ─────────────────────────────────

    def export_json(self) -> dict[str, Any]:
        """Plain-JSON weights (layer by layer) that docs/index.html can load."""
        layers = []
        for module in self.q_net.net:
            if isinstance(module, nn.Linear):
                layers.append(
                    {
                        "W": module.weight.detach().cpu().numpy().round(6).tolist(),
                        "b": module.bias.detach().cpu().numpy().round(6).tolist(),
                    }
                )
        return {
            "format": "rl_games-dqn-mlp",
            "version": 1,
            "env_id": self.env_id,
            "activation": "relu",
            "state_dim": self.state_dim,
            "action_dim": self.n_actions,
            "training_episodes": self.training_episodes,
            "layers": layers,
        }
