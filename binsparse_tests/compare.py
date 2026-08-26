"""Logical comparison of NPZ, Zarr, and HDF5 Binsparse containers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContainerContents:
    """The format-independent content relevant to a Binsparse container."""

    header: Mapping[str, Any]
    buffers: Mapping[str, np.ndarray]


class BinaryContainer(ABC):
    """Read the Binsparse header and declared buffers from one container."""

    def __init__(self, path: str | Path, source: Any) -> None:
        self.path = Path(path)
        self.source = source

    @abstractmethod
    def read_header(self) -> dict[str, Any]:
        """Return the decoded Binsparse header."""

    @abstractmethod
    def read_buffer(self, name: str) -> np.ndarray:
        """Return one encoded Binsparse buffer."""

    def read_contents(self) -> ContainerContents:
        """Return all format-independent Binsparse content."""
        header = self.read_header()
        data_types = header.get("data_types")
        if not isinstance(data_types, dict):
            raise AssertionError(f"{self.path} has invalid or missing data_types")
        try:
            buffers = {name: self.read_buffer(name) for name in data_types}
        except KeyError as error:
            raise AssertionError(
                f"{self.path} is missing declared buffer {error.args[0]!r}"
            ) from error
        return ContainerContents(header=header, buffers=buffers)


class NPZContainer(BinaryContainer):
    """Read a NumPy ZIP container."""

    def read_header(self) -> dict[str, Any]:
        return _decode_header(self.source["binsparse"])

    def read_buffer(self, name: str) -> np.ndarray:
        return np.asarray(self.source[name])


class HDF5Container(BinaryContainer):
    """Read an HDF5 container."""

    def read_header(self) -> dict[str, Any]:
        return _decode_header(self.source.attrs["binsparse"])

    def read_buffer(self, name: str) -> np.ndarray:
        return np.asarray(self.source[name])


class ZarrContainer(BinaryContainer):
    """Read a Zarr container."""

    def read_header(self) -> dict[str, Any]:
        return _decode_header(self.source.attrs["binsparse"])

    def read_buffer(self, name: str) -> np.ndarray:
        return np.asarray(self.source[name])


def assert_containers_equal(
    actual: str | Path,
    expected: str | Path,
    *,
    label: str = "Binsparse container",
) -> None:
    """Assert that two supported containers have equal headers and buffers."""
    actual_contents = read_container(actual)
    expected_contents = read_container(expected)

    if actual_contents.header != expected_contents.header:
        difference = _header_diff(actual_contents.header, expected_contents.header)
        raise AssertionError(f"{label} header differs:\n{difference}")

    for name in expected_contents.buffers:
        _assert_array_equal(
            f"{label} buffer {name!r}",
            actual_contents.buffers[name],
            expected_contents.buffers[name],
        )


def read_container(path: str | Path) -> ContainerContents:
    """Read the relevant content of an NPZ, Zarr, or HDF5 container."""
    with open_container(path) as container:
        return container.read_contents()


@contextmanager
def open_container(path: str | Path) -> Iterator[BinaryContainer]:
    """Open a supported backend and yield its Binsparse adapter."""
    container_path = Path(path)
    match container_path.suffix.lower():
        case ".npz":
            with np.load(container_path, allow_pickle=False) as archive:
                yield NPZContainer(container_path, archive)
        case ".h5" | ".hdf5":
            import h5py

            with h5py.File(container_path, "r") as file:
                yield HDF5Container(container_path, file)
        case ".zarr":
            import zarr

            group = zarr.open_group(container_path, mode="r")
            yield ZarrContainer(container_path, group)
        case _:
            raise ValueError(
                f"unsupported container extension {container_path.suffix!r}"
            )


def _decode_header(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise AssertionError("Binsparse header is not a JSON object")
    return value


def _header_diff(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> str:
    actual_lines = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    expected_lines = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    return "\n".join(
        unified_diff(
            expected_lines,
            actual_lines,
            fromfile="expected header",
            tofile="actual header",
            lineterm="",
        )
    )


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


__all__ = [
    "BinaryContainer",
    "ContainerContents",
    "HDF5Container",
    "NPZContainer",
    "ZarrContainer",
    "assert_containers_equal",
    "open_container",
    "read_container",
]
