from pathlib import Path

import pytest

from .run import EXECUTABLE_ENV, _resolve_commands


@pytest.fixture(autouse=True)
def clear_executable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINSPARSE_BIN", raising=False)
    for variable in EXECUTABLE_ENV.values():
        monkeypatch.delenv(variable, raising=False)


def test_binsparse_bin_supplies_executable_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BINSPARSE_BIN", str(tmp_path))
    commands = _resolve_commands(
        npy_to_binsparse=None,
        binsparse_to_npy=None,
        binsparse_to_binsparse=None,
    )

    assert commands == {
        executable: str(tmp_path / executable) for executable in EXECUTABLE_ENV
    }


def test_individual_variable_overrides_binsparse_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BINSPARSE_BIN", str(tmp_path))
    monkeypatch.setenv("BINSPARSE_TO_NPY", "/custom/binsparse_to_npy")
    commands = _resolve_commands(
        npy_to_binsparse=None,
        binsparse_to_npy=None,
        binsparse_to_binsparse=None,
    )

    assert commands["binsparse_to_npy"] == "/custom/binsparse_to_npy"
