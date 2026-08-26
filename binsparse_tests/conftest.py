from __future__ import annotations

import pytest


@pytest.fixture(
    params=[
        pytest.param(".npz", id="npz", marks=pytest.mark.npz),
        pytest.param(".zarr", id="zarr", marks=pytest.mark.zarr),
        pytest.param(".h5", id="hdf5", marks=pytest.mark.hdf5),
    ]
)
def container_suffix(request: pytest.FixtureRequest) -> str:
    """Return the suffix for each supported binary container."""
    return request.param
