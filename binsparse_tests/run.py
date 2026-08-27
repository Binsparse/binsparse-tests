from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import chdir
from pathlib import Path
from typing import Any

import numpy as np
from reference_cli import converters as reference_cli

from .compare import assert_containers_equal

Command = str | Sequence[str]
GeneratedInput = tuple[np.ndarray, np.ndarray, Any, Mapping[str, Any]]

EXECUTABLE_ENV = {
    "npy_to_binsparse": "NPY_TO_BINSPARSE",
    "binsparse_to_npy": "BINSPARSE_TO_NPY",
    "binsparse_to_binsparse": "BINSPARSE_TO_BINSPARSE",
}

REFERENCE_EXECUTABLES = {
    "npy_to_binsparse": reference_cli.npy_to_binsparse_main,
    "binsparse_to_npy": reference_cli.binsparse_to_npy_main,
    "binsparse_to_binsparse": reference_cli.binsparse_to_binsparse_main,
}


def run_executables(
    generated: GeneratedInput,
    *,
    npy_to_binsparse: Command | None = None,
    binsparse_to_npy: Command | None = None,
    binsparse_to_binsparse: Command | None = None,
    container_suffix: str = ".npz",
    workdir: str | os.PathLike[str] | None = None,
) -> None:
    commands = _resolve_commands(
        npy_to_binsparse=npy_to_binsparse,
        binsparse_to_npy=binsparse_to_npy,
        binsparse_to_binsparse=binsparse_to_binsparse,
    )

    tmp_parent = None if workdir is None else Path(workdir)
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_parent) as directory:
        _run_case(generated, commands, Path(directory), container_suffix)


def _run_case(
    generated: GeneratedInput,
    commands: Mapping[str, Command],
    directory: Path,
    container_suffix: str,
) -> None:
    tensor_in, pattern_in, fill_value_in, header_in = _write_npy_inputs(
        directory / "input",
        generated,
    )

    expected_from_npy = _container_path(
        directory,
        "expected_npy_to_binsparse",
        container_suffix,
    )
    actual_from_npy = _container_path(
        directory,
        "actual_npy_to_binsparse",
        container_suffix,
    )

    npy_to_binsparse_args = [
        tensor_in,
        pattern_in,
        fill_value_in,
        header_in,
        expected_from_npy,
    ]
    _run_reference("npy_to_binsparse", npy_to_binsparse_args, directory)
    _run_command(
        "npy_to_binsparse",
        commands["npy_to_binsparse"],
        [*npy_to_binsparse_args[:-1], actual_from_npy],
        directory,
    )
    _assert_binsparse_equal(
        "npy_to_binsparse",
        actual_from_npy,
        expected_from_npy,
    )

    expected_npy = _npy_paths(directory / "expected_binsparse_to_npy")
    actual_npy = _npy_paths(directory / "actual_binsparse_to_npy")
    _run_reference("binsparse_to_npy", [expected_from_npy, *expected_npy], directory)
    _run_command(
        "binsparse_to_npy",
        commands["binsparse_to_npy"],
        [expected_from_npy, *actual_npy],
        directory,
    )
    _assert_npy_equal("binsparse_to_npy", actual_npy, expected_npy)

    expected_roundtrip = _container_path(
        directory,
        "expected_binsparse_to_binsparse",
        container_suffix,
    )
    actual_roundtrip = _container_path(
        directory,
        "actual_binsparse_to_binsparse",
        container_suffix,
    )
    _run_reference(
        "binsparse_to_binsparse",
        [expected_from_npy, expected_roundtrip],
        directory,
    )
    _run_command(
        "binsparse_to_binsparse",
        commands["binsparse_to_binsparse"],
        [expected_from_npy, actual_roundtrip],
        directory,
    )
    _assert_binsparse_equal(
        "binsparse_to_binsparse",
        actual_roundtrip,
        expected_roundtrip,
    )


def _resolve_commands(
    *,
    npy_to_binsparse: Command | None,
    binsparse_to_npy: Command | None,
    binsparse_to_binsparse: Command | None,
) -> dict[str, Command]:
    commands = {
        "npy_to_binsparse": npy_to_binsparse
        or _configured_executable("npy_to_binsparse"),
        "binsparse_to_npy": binsparse_to_npy
        or _configured_executable("binsparse_to_npy"),
        "binsparse_to_binsparse": binsparse_to_binsparse
        or _configured_executable("binsparse_to_binsparse"),
    }
    missing = [
        EXECUTABLE_ENV[name] for name, command in commands.items() if command is None
    ]
    if missing:
        raise ValueError(
            f"missing executable environment variables: {', '.join(missing)}"
        )
    return {name: command for name, command in commands.items() if command is not None}


def _configured_executable(name: str) -> str | None:
    if command := os.environ.get(EXECUTABLE_ENV[name]):
        return command
    if executable_directory := os.environ.get("BINSPARSE_BIN"):
        return str(Path(executable_directory) / name)
    return None


def _write_npy_inputs(
    prefix: Path,
    generated: GeneratedInput,
) -> tuple[Path, Path, Path, Path]:
    dense, pattern, fill_value, header = generated
    tensor_in = prefix.with_name(f"{prefix.name}_tensor.npy")
    pattern_in = prefix.with_name(f"{prefix.name}_pattern.npy")
    fill_value_in = prefix.with_name(f"{prefix.name}_fill_value.npy")
    header_in = prefix.with_name(f"{prefix.name}_header.json")

    np.save(tensor_in, dense)
    np.save(pattern_in, pattern)
    np.save(fill_value_in, np.asarray(fill_value))
    with header_in.open("w", encoding="utf-8") as file:
        json.dump(header, file, indent=2, sort_keys=True)

    return tensor_in, pattern_in, fill_value_in, header_in


def _run_command(
    name: str,
    command: Command,
    args: Sequence[Path],
    cwd: Path,
) -> None:
    argv = [*_command_parts(command), *[str(arg) for arg in args]]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            cwd=cwd,
            text=True,
        )
    except FileNotFoundError as error:
        raise AssertionError(f"{name} executable was not found: {argv[0]}") from error
    if completed.returncode != 0:
        raise AssertionError(
            f"{name} exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _run_reference(name: str, args: Sequence[Path], cwd: Path) -> None:
    with chdir(cwd):
        exit_code = REFERENCE_EXECUTABLES[name]([str(arg) for arg in args])
    if exit_code != 0:
        raise AssertionError(f"reference {name} exited with {exit_code}")


def _assert_binsparse_equal(
    name: str,
    actual: Path,
    expected: Path,
) -> None:
    assert_containers_equal(actual, expected, label=name)


def _assert_npy_equal(
    name: str,
    actual: tuple[Path, Path, Path],
    expected: tuple[Path, Path, Path],
) -> None:
    for label, actual_path, expected_path in zip(
        ("tensor", "pattern", "fill_value"),
        actual,
        expected,
        strict=True,
    ):
        actual_array = np.load(actual_path, allow_pickle=False)
        expected_array = np.load(expected_path, allow_pickle=False)
        _assert_array_equal(f"{name} {label}", actual_array, expected_array)


def _assert_array_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: {actual.dtype!s} != {expected.dtype!s}"
        )
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: {actual.shape} != {expected.shape}"
        )
    if not np.array_equal(actual, expected, equal_nan=True):
        raise AssertionError(f"{name} values differ")


def _command_parts(command: Command) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]
    path_separators = tuple(
        separator for separator in (os.sep, os.altsep) if separator is not None
    )
    if parts and any(separator in parts[0] for separator in path_separators):
        parts[0] = str(Path(parts[0]).expanduser().resolve())
    return parts


def _container_path(directory: Path, name: str, suffix: str) -> Path:
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return directory / f"{name}{suffix}"


def _npy_paths(prefix: Path) -> tuple[Path, Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return (
        prefix.with_name(f"{prefix.name}_tensor.npy"),
        prefix.with_name(f"{prefix.name}_pattern.npy"),
        prefix.with_name(f"{prefix.name}_fill_value.npy"),
    )


__all__ = ["run_executables"]
