"""Experience replay buffer backed by pre-allocated NumPy arrays.

Contents:
  - Batch        : a sampled mini-batch as NumPy arrays
  - ReplayBuffer : fixed-size circular buffer of (s, a, r, s', terminated)

Versus a deque of tuples, contiguous arrays make `sample()` a single fancy
index per field (no Python loop, no np.stack of 128 small arrays), use a
fixed amount of memory, and sample through the agent's own seeded RNG.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    states: np.ndarray  # (B, obs_dim) float32
    actions: np.ndarray  # (B,) int64
    rewards: np.ndarray  # (B,) float32
    next_states: np.ndarray  # (B, obs_dim) float32
    terminated: np.ndarray  # (B,) float32 -- 1.0 only for true terminal states


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} from {self._size} transitions")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminated[idx],
        )

    def __len__(self) -> int:
        return self._size
