from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .generate import (
    boolean_value_datatypes,
    complex_datatypes,
    floating_datatypes,
    iso_datatypes,
    npy_inputs,
    signed_integer_datatypes,
    unsigned_integer_datatypes,
)
from .run import run_executables

MAX_EXAMPLES = 5

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
    generated = data.draw(npy_inputs(n=1, values_dtypes=values_dtypes), label="generated")
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
    generated = data.draw(npy_inputs(n=2, values_dtypes=values_dtypes), label="generated")
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
        npy_inputs(format_name="CSR", values_dtypes=values_dtypes),
        label="generated",
    )
    run_executables(generated)
