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
    signed_integer_datatypes,
    unsigned_integer_datatypes,
)
from .run import run_executables

MAX_EXAMPLES = 5
DIMENSION = st.integers(min_value=0, max_value=4)
SHAPE_1D = st.tuples(DIMENSION)
SHAPE_2D = st.tuples(DIMENSION, DIMENSION)

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


@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_1_dim(
    data: st.DataObject, values_dtypes: st.SearchStrategy[str], iso: bool
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    generated = data.draw(
        npy_inputs(
            shape=SHAPE_1D,
            format=formats(1),
            datatypes=datatypes(values_dtypes),
            transpose=optional_transposes(1),
        ),
        label="generated",
    )
    run_executables(generated)


@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_2_dim(
    data: st.DataObject, values_dtypes: st.SearchStrategy[str], iso: bool
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    generated = data.draw(
        npy_inputs(
            shape=SHAPE_2D,
            format=formats(2),
            datatypes=datatypes(values_dtypes),
            transpose=optional_transposes(2),
        ),
        label="generated",
    )
    run_executables(generated)


@pytest.mark.parametrize("iso", ISO)
@pytest.mark.parametrize("values_dtypes", VALUE_DTYPE_STRATEGIES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_csr(
    data: st.DataObject, values_dtypes: st.SearchStrategy[str], iso: bool
) -> None:
    if iso:
        values_dtypes = iso_datatypes(values_dtypes)
    generated = data.draw(
        npy_inputs(
            shape=SHAPE_2D,
            format=st.just(predefined["CSR"][0]),
            format_name=st.just("CSR"),
            datatypes=datatypes(values_dtypes),
        ),
        label="generated",
    )
    run_executables(generated)
