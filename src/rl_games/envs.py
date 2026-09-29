"""Environment construction and observation bounds.

Contents:
  - DEFAULT_ENV_ID : the env used when none is given
  - OBS_BOUNDS     : hand-written observation ranges, per env id
  - make()         : create a Gymnasium environment (optionally seeded)
  - bounds_for()   : observation bounds for an env, hand-written or derived
  - check_discrete_actions(): fail early on continuous-action envs
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

DEFAULT_ENV_ID = "LunarLander-v3"

# env id -> (bounds for the leading continuous dims, count of trailing dims
# that are already discrete and so are used as bin indices verbatim).
OBS_BOUNDS: dict[str, tuple[np.ndarray, int]] = {
    # 0-5 continuous (x, y, vx, vy, angle, angular velocity); 6-7 leg-contact flags
    "LunarLander-v3": (
        np.array(
            [
                [-1.5, 1.5],
                [-0.5, 1.5],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-3.14, 3.14],
                [-5.0, 5.0],
            ]
        ),
        2,
    ),
    # cart position, cart velocity, pole angle (rad), pole angular velocity
    "CartPole-v1": (
        np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.21, 0.21], [-3.0, 3.0]]),
        0,
    ),
}


def make(
    env_id: str | None = None,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Create an environment, defaulting to this project's primary env.

    When `seed` is given the action space is seeded too, so that
    `env.action_space.sample()` is reproducible as well.
    """
    env = gym.make(env_id or DEFAULT_ENV_ID, render_mode=render_mode)
    check_discrete_actions(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def check_discrete_actions(env: gym.Env) -> None:
    """Both agents output an action index, so the space must be Discrete."""
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env_id = env.spec.id if env.spec else "this environment"
        raise ValueError(
            f"{env_id} has a {type(env.action_space).__name__} action space. "
            "Q-learning and DQN need a Discrete one (e.g. CartPole-v1, "
            "LunarLander-v3, Acrobot-v1, MountainCar-v0)."
        )


def bounds_for(env_id: str, env: gym.Env) -> tuple[np.ndarray, int]:
    """Observation bounds for `env_id`: hand-written if listed, else the env's own.

    Raises:
        ValueError: if the env reports an unbounded space and has no entry.
    """
    if env_id in OBS_BOUNDS:
        return OBS_BOUNDS[env_id]

    space = env.observation_space
    low = np.asarray(getattr(space, "low", np.array([])), dtype=float)
    high = np.asarray(getattr(space, "high", np.array([])), dtype=float)
    if low.size == 0 or not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(
            f"{env_id!r} reports an unbounded observation space, so a tabular "
            "agent cannot pick bin edges for it. Add an OBS_BOUNDS entry in "
            "src/rl_games/envs.py, or use a DQN agent, which needs none."
        )
    return np.stack([low, high], axis=1), 0
