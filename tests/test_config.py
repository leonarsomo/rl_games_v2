from __future__ import annotations

import pytest

from rl_games.config import (
    DQNConfig,
    QLearningConfig,
    from_dict,
    make_config,
    parse_overrides,
    to_dict,
)


def test_preset_then_override_precedence() -> None:
    cfg = make_config("dqn", "CartPole-v1", {"lr": 0.01})
    assert isinstance(cfg, DQNConfig)
    assert cfg.lr == 0.01  # override wins
    assert cfg.epsilon_decay_steps == 10_000  # from preset
    assert cfg.double_dqn is True  # class default


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        make_config("qlearning", "CartPole-v1", {"learning_rate": 0.1})


@pytest.mark.parametrize(
    "kwargs",
    [{"gamma": 1.5}, {"epsilon_end": 0.5, "epsilon_start": 0.1}, {"n_bins": 1}],
)
def test_invalid_values_fail_fast(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"must|need"):
        QLearningConfig(**kwargs)


def test_parse_overrides_types() -> None:
    out = parse_overrides(
        "dqn", ["lr=3e-4", "hidden=64,32", "double_dqn=false", "buffer_capacity=1e5"]
    )
    assert out == {
        "lr": 3e-4,
        "hidden": (64, 32),
        "double_dqn": False,
        "buffer_capacity": 100_000,
    }


def test_parse_overrides_bad_key() -> None:
    with pytest.raises(ValueError, match="Bad override"):
        parse_overrides("dqn", ["nope=1"])


def test_roundtrip_dict() -> None:
    cfg = DQNConfig(hidden=(32, 16))
    assert from_dict("dqn", to_dict(cfg)) == cfg
