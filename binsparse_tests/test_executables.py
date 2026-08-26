from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .generate import (
    boolean_value_datatypes,
    complex_datatypes,
    datatypes,
    floating_datatypes,
    formats,
    iso_datatypes,
    npy_inputs,
    optional_transposes,
    predefined,
    predefined_1d,
    predefined_2d,
    signed_integer_datatypes,
    unsigned_integer_datatypes,
)
from .run import run_executables

MAX_EXAMPLES = 5
DIMENSION = st.integers(min_value=0, max_value=4)
SHAPE_0D = st.just(())
SHAPE_1D = st.tuples(DIMENSION)
SHAPE_2D = st.tuples(DIMENSION, DIMENSION)
SHAPE_3D = st.tuples(DIMENSION, DIMENSION, DIMENSION)
NDIM = st.integers(min_value=4, max_value=6)

VALUE_DTYPE_STRATEGIES = [
    pytest.param(
        boolean_value_datatypes(), id="boolean", marks=pytest.mark.bool_values
    ),
    pytest.param(
        signed_integer_datatypes(), id="signed", marks=pytest.mark.signed_integer
    ),
    pytest.param(
        unsigned_integer_datatypes(),
        id="unsigned",
        marks=pytest.mark.unsigned_integer,
    ),
    pytest.param(floating_datatypes(), id="floating", marks=pytest.mark.floating),
    pytest.param(
        complex_datatypes(),
        id="complex",
        marks=pytest.mark.complex_values,
    ),
]

ISO = [
    pytest.param(False, id="non-iso"),
    pytest.param(True, id="iso", marks=pytest.mark.iso),
]

CONTAINERS = [
    pytest.param(".npz", id="npz", marks=pytest.mark.npz),
    pytest.param(".zarr", id="zarr", marks=pytest.mark.zarr),
    pytest.param(".h5", id="hdf5", marks=pytest.mark.hdf5),
]

PREDEFINED_1D = [pytest.param(name, id=name) for name in predefined_1d]
PREDEFINED_2D = [pytest.param(name, id=name) for name in predefined_2d]


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_0d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    _run_custom_case(
        data,
        ndim=0,
        shape=SHAPE_0D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_1d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    _run_custom_case(
        data,
        ndim=1,
        shape=SHAPE_1D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_2d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    _run_custom_case(
        data,
        ndim=2,
        shape=SHAPE_2D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_3d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    _run_custom_case(
        data,
        ndim=3,
        shape=SHAPE_3D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data(), ndim=NDIM)
def test_custom_nd(
    data: st.DataObject,
    ndim: int,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    _run_custom_case(
        data,
        ndim=ndim,
        shape=st.tuples(*([DIMENSION] * ndim)),
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
    )


def _run_custom_case(
    data: st.DataObject,
    *,
    ndim: int,
    shape: st.SearchStrategy[tuple[int, ...]],
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    generated = data.draw(
        npy_inputs(
            shape=shape,
            format=formats(ndim),
            datatypes=datatypes(values_dtypes),
            transpose=optional_transposes(ndim),
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@pytest.mark.parametrize("format_name", PREDEFINED_1D)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_predefined_1d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    format_name: str,
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    format, transpose = predefined[format_name]
    generated = data.draw(
        npy_inputs(
            shape=SHAPE_1D,
            format=st.just(format),
            format_name=st.just(format_name),
            datatypes=datatypes(values_dtypes),
            transpose=st.just(transpose),
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@pytest.mark.parametrize("format_name", PREDEFINED_2D)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_predefined_2d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    format_name: str,
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    format, transpose = predefined[format_name]
    generated = data.draw(
        npy_inputs(
            shape=SHAPE_2D,
            format=st.just(format),
            format_name=st.just(format_name),
            datatypes=datatypes(values_dtypes),
            transpose=st.just(transpose),
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)
