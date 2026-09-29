from __future__ import annotations

from pathlib import Path

import pytest

from rl_games import registry
from rl_games.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "rl_games" in capsys.readouterr().out


def test_train_load_sim_delete_cycle(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--env", "CartPole-v1"]
    assert main(["train", "qlearning", *args, "--episodes", "20", "--seed", "0"]) == 0
    assert registry.save_path("qlearning", "CartPole-v1").exists()
    assert registry.log_path("qlearning", "CartPole-v1").exists()
    assert main(["load", "qlearning", *args, "--eval", "--episodes", "2"]) == 0
    assert main(["sim", "qlearning", *args, "--steps", "3"]) == 0
    assert main(["delete", "qlearning", *args]) == 0
    assert not registry.save_path("qlearning", "CartPole-v1").exists()


def test_overrides_refused_on_existing_save() -> None:
    args = ["train", "qlearning", "--env", "CartPole-v1", "--episodes", "2"]
    assert main(args) == 0
    assert main([*args, "--hp", "lr=0.5"]) == 2


def test_bad_override_gives_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["train", "dqn", "--env", "CartPole-v1", "--hp", "nope=1"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_continuous_action_env_is_rejected() -> None:
    assert main(["train", "dqn", "--env", "Pendulum-v1", "--episodes", "1"]) == 2


def test_export(tmp_path: Path) -> None:
    assert (
        main(
            [
                "train",
                "dqn",
                "--env",
                "CartPole-v1",
                "--episodes",
                "2",
                "--hp",
                "learning_starts=10",
                "--hp",
                "batch_size=8",
            ]
        )
        == 0
    )
    out = tmp_path / "m.json"
    assert main(["export", "dqn", "--env", "CartPole-v1", "--out", str(out)]) == 0
    assert out.stat().st_size > 1000
