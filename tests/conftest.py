"""Shared fixtures. Every test writes saves into a temporary directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_games import registry


@pytest.fixture(autouse=True)
def isolated_save_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(registry, "SAVE_DIR", tmp_path / "saves")
    return tmp_path / "saves"
