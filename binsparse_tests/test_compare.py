from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .compare import assert_containers_equal, read_container

HEADER = {"format": "CSR", "data_types": {"values": "float64"}}
VALUES = np.asarray([1.0, np.nan, 3.0])


def _write_npz(path: Path, values: np.ndarray = VALUES) -> None:
    np.savez(path, binsparse=json.dumps({"binsparse": HEADER}), values=values)


def test_npz_headers_compare_as_json(tmp_path: Path) -> None:
    actual = tmp_path / "actual.npz"
    expected = tmp_path / "expected.npz"
    np.savez(actual, binsparse=json.dumps({"binsparse": HEADER}), values=VALUES)
    np.savez(
        expected,
        binsparse=json.dumps(
            {
                "binsparse": {
                    "data_types": HEADER["data_types"],
                    "format": "CSR",
                }
            }
        ),
        values=VALUES,
    )

    assert_containers_equal(actual, expected)


def test_buffer_difference_is_reported(tmp_path: Path) -> None:
    actual = tmp_path / "actual.npz"
    expected = tmp_path / "expected.npz"
    _write_npz(actual, np.asarray([1.0, 2.0]))
    _write_npz(expected, np.asarray([1.0, 3.0]))

    with pytest.raises(AssertionError, match="values differ"):
        assert_containers_equal(actual, expected)


@pytest.mark.parametrize("data_type", ["bint8", "iso[bint8]"])
def test_bint8_compares_as_numpy_boolean(tmp_path: Path, data_type: str) -> None:
    h5py = pytest.importorskip("h5py")
    header = {"format": "CSR", "data_types": {"values": data_type}}
    actual = tmp_path / "actual.h5"
    expected = tmp_path / "expected.npz"
    with h5py.File(actual, "w") as file:
        file.attrs["binsparse"] = json.dumps({"binsparse": header})
        file.create_dataset("values", data=np.asarray([0, 1], dtype=np.uint8))
    np.savez(
        expected,
        binsparse=json.dumps({"binsparse": header}),
        values=np.asarray([False, True]),
    )

    assert_containers_equal(actual, expected)


def test_non_bint8_still_requires_matching_dtype(tmp_path: Path) -> None:
    header = {"format": "CSR", "data_types": {"values": "uint8"}}
    actual = tmp_path / "actual.npz"
    expected = tmp_path / "expected.npz"
    np.savez(
        actual,
        binsparse=json.dumps({"binsparse": header}),
        values=np.asarray([0, 1], dtype=np.uint8),
    )
    np.savez(
        expected,
        binsparse=json.dumps({"binsparse": header}),
        values=np.asarray([0, 1], dtype=np.int8),
    )

    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_containers_equal(actual, expected)


def test_header_difference_shows_json_diff(tmp_path: Path) -> None:
    actual = tmp_path / "actual.npz"
    expected = tmp_path / "expected.npz"
    actual_header = {**HEADER, "format": "CSC"}
    np.savez(actual, binsparse=json.dumps({"binsparse": actual_header}), values=VALUES)
    _write_npz(expected)

    with pytest.raises(AssertionError) as raised:
        assert_containers_equal(actual, expected, label="roundtrip")

    message = str(raised.value)
    assert message.startswith("roundtrip header differs:\n--- expected header")
    assert '  "format": "CSR"' in message
    assert '  "format": "CSC"' in message


def test_hdf5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tensor.h5"
    with h5py.File(path, "w") as file:
        file.attrs["binsparse"] = json.dumps({"binsparse": HEADER})
        file.create_dataset("values", data=VALUES)

    assert np.array_equal(
        read_container(path).buffers["values"], VALUES, equal_nan=True
    )


def test_zarr(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tensor.zarr"
    group = zarr.open_group(path, mode="w")
    group.attrs["binsparse"] = {"binsparse": HEADER}
    group.create_array("values", data=VALUES)

    assert np.array_equal(
        read_container(path).buffers["values"], VALUES, equal_nan=True
    )
