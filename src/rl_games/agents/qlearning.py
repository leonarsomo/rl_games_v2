"""Tabular Q-learning over a discretised observation space.

Contents:
  - QLearningAgent : discretisation, greedy lookup, TD(0) update,
                     per-episode ε decay, and a pickle-free save format.

Update rule (Watkins & Dayan, 1992):

    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') · (1 − terminated) − Q(s,a) ]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Self

import numpy as np

from rl_games import envs
from rl_games.agents.base import BaseAgent
from rl_games.config import QLearningConfig, from_dict

State = tuple[int, ...]


class QLearningAgent(BaseAgent[State, QLearningConfig]):
    """Tabular Q-learning over a discretised observation space."""

    agent_type = "qlearning"
    label = "Q-Learning"

    def __init__(
        self, env_id: str, config: QLearningConfig | None = None, *, seed: int | None = None
    ) -> None:
        super().__init__(env_id, config or QLearningConfig(), seed=seed)
        self.q_table: defaultdict[State, np.ndarray] = self._empty_table()

    def _setup(self, env: Any) -> None:
        self._bounds, self._n_binary_dims = envs.bounds_for(self.env_id, env)
        self._low = self._bounds[:, 0]
        self._high = self._bounds[:, 1]
        # n_bins intervals need n_bins - 1 *interior* edges.
        self._edges = [np.linspace(lo, hi, self.config.n_bins + 1)[1:-1] for lo, hi in self._bounds]
        self._n_continuous = len(self._bounds)

    def _empty_table(self) -> defaultdict[State, np.ndarray]:
        n = self.n_actions
        return defaultdict(lambda: np.zeros(n))

    # ── state representation ───────────────────────────────────────────

    def discretize(self, obs: np.ndarray) -> State:
        """Map a continuous observation to a hashable tuple of bin indices.

        Values outside [low, high] are clipped so they land in the edge bins
        instead of creating unbounded new keys.
        """
        obs = np.asarray(obs, dtype=float)
        cont = np.clip(obs[: self._n_continuous], self._low, self._high)
        bins = [int(np.digitize(v, edges)) for v, edges in zip(cont, self._edges, strict=True)]
        flags = [round(float(v)) for v in obs[self._n_continuous :]]
        return tuple(bins + flags)

    def _to_state(self, obs: np.ndarray) -> State:
        return self.discretize(obs)

    # ── policy & learning ──────────────────────────────────────────────

    def _greedy_action(self, state: State) -> int:
        q = self.q_table[state]
        # Break ties randomly: with an all-zero row np.argmax would always
        # return action 0, biasing early behaviour.
        best = np.flatnonzero(q == q.max())
        return int(best[0] if best.size == 1 else self.rng.choice(best))

    def _observe(
        self, state: State, action: int, reward: float, next_state: State, terminated: bool
    ) -> float:
        future = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        target = reward + self.config.gamma * future
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.config.lr * td_error
        return float(td_error**2)  # reported as "loss" for comparability with DQN

    def _on_episode_end(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    # ── persistence (npz + JSON header, no pickle) ─────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.q_table.keys()), dtype=np.int32).reshape(len(self.q_table), -1)
        values = (
            np.stack(list(self.q_table.values())) if self.q_table else np.zeros((0, self.n_actions))
        )
        meta = json.dumps(self._common_state())
        with path.open("wb") as f:
            np.savez_compressed(f, keys=keys, values=values, meta=np.array(meta))
        print(f"Saved {self.label} agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            keys, values = data["keys"], data["values"]
        cls._check_format(meta, path)
        cfg = from_dict(cls.agent_type, meta["config"])
        assert isinstance(cfg, QLearningConfig)
        agent = cls(meta["env_id"], cfg, seed=meta.get("seed"))
        agent._restore_common(meta)
        for k, v in zip(keys, values, strict=True):
            agent.q_table[tuple(int(x) for x in k)] = v.astype(float)
        return agent

    def _info_extra(self) -> list[str]:
        return [f"  States visited   : {len(self.q_table)}"]

    def _progress_extra(self) -> str:
        return f" | States: {len(self.q_table)}"
