"""Common base class for the agents.

Contents:
  - BaseAgent : shared state (config, ε, counters, RNG), the ε-greedy policy
                skeleton, ONE shared training loop with hooks, greedy
                evaluation during training, and the persistence contract.

Why a single training loop?  In the original project each agent copied the
same environment loop. Here the loop lives once, and each algorithm only
says what happens after a transition (`_observe`) and at episode end
(`_on_episode_end`). That keeps the algorithms short and the loop correct in
one place -- in particular the terminated / truncated distinction below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

import numpy as np

from rl_games import envs
from rl_games.config import AgentConfig
from rl_games.metrics import EpisodeRecord, TrainingLog
from rl_games.utils import make_rng

S = TypeVar("S")  # the agent's internal state representation
C = TypeVar("C", bound=AgentConfig)

SAVE_FORMAT_VERSION = 2

EpisodeCallback = Callable[[EpisodeRecord], None]


class BaseAgent(ABC, Generic[S, C]):
    """Common base for the tabular and neural agents."""

    #: Registry key ("qlearning", "dqn") and human-readable name.
    agent_type: ClassVar[str] = "base"
    label: ClassVar[str] = "Agent"

    def __init__(self, env_id: str, config: C, *, seed: int | None = None) -> None:
        self.env_id = env_id
        self.config: C = config
        self.seed = seed
        self.rng = make_rng(seed)
        self.epsilon: float = config.epsilon_start
        self.training_episodes = 0
        self.total_steps = 0

        probe = envs.make(env_id)
        try:
            self.n_actions = int(probe.action_space.n)  # type: ignore[attr-defined]
            self._setup(probe)
        finally:
            probe.close()

    # ── hooks each algorithm implements ────────────────────────────────

    def _setup(self, env: Any) -> None:
        """Read whatever the agent needs from a freshly-made env."""

    @abstractmethod
    def _to_state(self, obs: np.ndarray) -> S:
        """Convert a raw observation into the agent's state representation."""

    @abstractmethod
    def _greedy_action(self, state: S) -> int:
        """argmax_a Q(state, a)."""

    @abstractmethod
    def _observe(
        self, state: S, action: int, reward: float, next_state: S, terminated: bool
    ) -> float | None:
        """Learn from one transition. Returns a loss value when an update ran."""

    def _on_step_end(self) -> None:
        """Called after every environment step (e.g. per-step ε, target sync)."""

    def _on_episode_end(self) -> None:
        """Called after every training episode (e.g. per-episode ε decay)."""

    # ── policy ─────────────────────────────────────────────────────────

    def select_action(self, state: S, *, deterministic: bool = False) -> int:
        """ε-greedy: random with probability ε (unless deterministic), else greedy."""
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(state)

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        """Choose an action for a raw observation. Returns (action, None) like SB3."""
        return self.select_action(self._to_state(obs), deterministic=deterministic), None

    # ── training ───────────────────────────────────────────────────────

    def train(
        self,
        total_episodes: int,
        *,
        log_interval: int | None = None,
        eval_every: int = 0,
        eval_episodes: int = 5,
        callback: EpisodeCallback | None = None,
        eval_callback: Callable[[float], None] | None = None,
        verbose: bool = True,
    ) -> TrainingLog:
        """Run `total_episodes` of ε-greedy interaction and learning.

        Terminated vs truncated -- the most common DQN bug:
        `terminated` means the MDP really ended (crash, landing, pole fell),
        so the future value is 0. `truncated` means we hit the time limit;
        the state still has a future, so we must keep bootstrapping from it.
        Only `terminated` is passed to the learner; both end the episode.
        """
        if total_episodes < 1:
            raise ValueError("total_episodes must be >= 1")
        log_interval = log_interval or max(1, total_episodes // 20)
        env = envs.make(self.env_id, seed=self._next_seed())
        log = TrainingLog()

        try:
            for episode in range(1, total_episodes + 1):
                obs, _ = env.reset(seed=self._next_seed())
                state = self._to_state(obs)
                ep_reward, ep_steps, losses = 0.0, 0, []
                done = False

                while not done:
                    action = self.select_action(state)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = self._to_state(next_obs)

                    loss = self._observe(state, action, float(reward), next_state, terminated)
                    if loss is not None:
                        losses.append(loss)

                    self.total_steps += 1
                    self._on_step_end()
                    state = next_state
                    ep_reward += float(reward)
                    ep_steps += 1
                    done = terminated or truncated

                self.training_episodes += 1
                self._on_episode_end()

                record = EpisodeRecord(
                    episode=self.training_episodes,
                    steps=ep_steps,
                    total_steps=self.total_steps,
                    reward=ep_reward,
                    epsilon=self.epsilon,
                    loss=float(np.mean(losses)) if losses else float("nan"),
                )
                log.add(record)
                if callback is not None:
                    callback(record)

                if verbose and episode % log_interval == 0:
                    self._print_progress(episode, total_episodes, log, log_interval)

                if eval_every and episode % eval_every == 0:
                    from rl_games.evaluate import run_episodes

                    returns = run_episodes(self, n_episodes=eval_episodes, seed=10_000)
                    if eval_callback is not None:
                        eval_callback(float(np.mean(returns)))
                    if verbose:
                        print(
                            f"  [eval] greedy mean over {eval_episodes} eps: "
                            f"{np.mean(returns):.2f} ± {np.std(returns):.2f}"
                        )
        finally:
            env.close()
        return log

    def _next_seed(self) -> int | None:
        """Derive per-episode seeds from the agent's RNG (None when unseeded)."""
        if self.seed is None:
            return None
        return int(self.rng.integers(2**31 - 1))

    def _print_progress(self, episode: int, total: int, log: TrainingLog, window: int) -> None:
        last = log.records[-1]
        loss = "" if np.isnan(last.loss) else f" | Loss: {last.loss:.4f}"
        print(
            f"Episode {episode:>6}/{total} | "
            f"Avg reward ({window}): {log.mean_reward(window):8.2f} | "
            f"ε: {self.epsilon:.3f} | Steps: {self.total_steps}{loss}"
            f"{self._progress_extra()}"
        )

    def _progress_extra(self) -> str:
        return ""

    # ── persistence ────────────────────────────────────────────────────

    def _common_state(self) -> dict[str, Any]:
        from rl_games.config import to_dict

        return {
            "format_version": SAVE_FORMAT_VERSION,
            "agent_type": self.agent_type,
            "env_id": self.env_id,
            "config": to_dict(self.config),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps,
            "seed": self.seed,
        }

    def _restore_common(self, data: dict[str, Any]) -> None:
        self.epsilon = float(data["epsilon"])
        self.training_episodes = int(data["training_episodes"])
        self.total_steps = int(data["total_steps"])

    @staticmethod
    def _check_format(data: dict[str, Any], path: Path) -> None:
        version = int(data.get("format_version", 1))
        if version != SAVE_FORMAT_VERSION:
            raise ValueError(
                f"{path} uses save format v{version}; this version reads "
                f"v{SAVE_FORMAT_VERSION}. Delete it and retrain."
            )

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the agent to `path`."""

    # @classmethod must stay the OUTER decorator; reversed, abstractness is lost.
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Restore an agent previously written by `save`."""

    def info(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            f"{self.label} agent for {self.env_id}",
            f"  Episodes trained : {self.training_episodes}",
            f"  Env steps        : {self.total_steps}",
            f"  Epsilon          : {self.epsilon:.4f}",
            "  Config           :",
        ]
        from rl_games.config import to_dict

        lines += [f"    {k:<20} = {v}" for k, v in to_dict(self.config).items()]
        lines += self._info_extra()
        return "\n".join(lines)

    def _info_extra(self) -> list[str]:
        return []
