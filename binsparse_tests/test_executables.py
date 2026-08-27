from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .generate import (
    boolean_value_datatypes,
    complex_datatypes,
    datatypes,
    dense,
    element,
    floating_datatypes,
    formats,
    iso_datatypes,
    npy_inputs,
    predefined,
    predefined_1d,
    predefined_2d,
    signed_integer_datatypes,
    sparse,
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

FILL_VALUE_KINDS = [
    pytest.param("zero", id="zero-fill"),
    pytest.param("nonzero", id="nonzero-fill"),
]

CUSTOM_FORMAT_KINDS = [
    pytest.param("dense", id="dense-layout"),
    pytest.param("coo", id="coo-layout"),
    pytest.param("mixed", id="mixed-layout"),
]

CUSTOM_TRANSPOSE_KINDS = [
    pytest.param("none", id="no-transpose"),
    pytest.param("permuted", id="transposed"),
]

PREDEFINED_1D = [pytest.param(name, id=name) for name in predefined_1d]
PREDEFINED_2D = [pytest.param(name, id=name) for name in predefined_2d]


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
@pytest.mark.parametrize("transpose_kind", CUSTOM_TRANSPOSE_KINDS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_0d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    fill_value_kind: str,
    transpose_kind: str,
) -> None:
    _run_custom_case(
        data,
        ndim=0,
        shape=SHAPE_0D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
        fill_value_kind=fill_value_kind,
        transpose_kind=transpose_kind,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
@pytest.mark.parametrize("format_kind", CUSTOM_FORMAT_KINDS)
@pytest.mark.parametrize("transpose_kind", CUSTOM_TRANSPOSE_KINDS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_1d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    fill_value_kind: str,
    format_kind: str,
    transpose_kind: str,
) -> None:
    _run_custom_case(
        data,
        ndim=1,
        shape=SHAPE_1D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
        fill_value_kind=fill_value_kind,
        format_kind=format_kind,
        transpose_kind=transpose_kind,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
@pytest.mark.parametrize("format_kind", CUSTOM_FORMAT_KINDS)
@pytest.mark.parametrize("transpose_kind", CUSTOM_TRANSPOSE_KINDS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_2d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    fill_value_kind: str,
    format_kind: str,
    transpose_kind: str,
) -> None:
    _run_custom_case(
        data,
        ndim=2,
        shape=SHAPE_2D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
        fill_value_kind=fill_value_kind,
        format_kind=format_kind,
        transpose_kind=transpose_kind,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
@pytest.mark.parametrize("format_kind", CUSTOM_FORMAT_KINDS)
@pytest.mark.parametrize("transpose_kind", CUSTOM_TRANSPOSE_KINDS)
@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_custom_3d(
    data: st.DataObject,
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    fill_value_kind: str,
    format_kind: str,
    transpose_kind: str,
) -> None:
    _run_custom_case(
        data,
        ndim=3,
        shape=SHAPE_3D,
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
        fill_value_kind=fill_value_kind,
        format_kind=format_kind,
        transpose_kind=transpose_kind,
    )


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
@pytest.mark.parametrize("format_kind", CUSTOM_FORMAT_KINDS)
@pytest.mark.parametrize("transpose_kind", CUSTOM_TRANSPOSE_KINDS)
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
    fill_value_kind: str,
    format_kind: str,
    transpose_kind: str,
) -> None:
    _run_custom_case(
        data,
        ndim=ndim,
        shape=st.tuples(*([DIMENSION] * ndim)),
        values_dtypes=values_dtypes,
        iso=iso,
        container_suffix=container_suffix,
        fill_value_kind=fill_value_kind,
        format_kind=format_kind,
        transpose_kind=transpose_kind,
    )


def _run_custom_case(
    data: st.DataObject,
    *,
    ndim: int,
    shape: st.SearchStrategy[tuple[int, ...]],
    values_dtypes: st.SearchStrategy[str],
    iso: bool,
    container_suffix: str,
    fill_value_kind: str,
    format_kind: str = "mixed",
    transpose_kind: str = "none",
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    generated = data.draw(
        npy_inputs(
            shape=shape,
            format=_custom_formats(ndim, format_kind),
            datatypes=datatypes(values_dtypes),
            transpose=_custom_transposes(ndim, transpose_kind),
            fill_value_kind=fill_value_kind,
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)


def _custom_formats(ndim: int, kind: str) -> st.SearchStrategy[dict]:
    if ndim == 0:
        return st.just(element)
    if kind == "dense":
        return st.just(dense(element, rank=ndim))
    if kind == "coo":
        return st.just(sparse(element, rank=ndim))
    if kind == "mixed":
        return formats(ndim)
    raise ValueError(f"unknown custom format kind {kind!r}")


def _custom_transposes(ndim: int, kind: str) -> st.SearchStrategy[tuple | None]:
    if kind == "none":
        return st.none()
    if kind == "permuted":
        return st.permutations(range(ndim)).map(tuple)
    raise ValueError(f"unknown custom transpose kind {kind!r}")


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
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
    fill_value_kind: str,
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
            fill_value_kind=fill_value_kind,
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)


@pytest.mark.parametrize("container_suffix", CONTAINERS)
@pytest.mark.parametrize("fill_value_kind", FILL_VALUE_KINDS)
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
    fill_value_kind: str,
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
            fill_value_kind=fill_value_kind,
        ),
        label="generated",
    )
    run_executables(generated, container_suffix=container_suffix)
