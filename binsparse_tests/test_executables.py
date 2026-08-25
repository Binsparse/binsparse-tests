from __future__ import annotations

import os
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .generate import npy_inputs
from .run import EXECUTABLE_ENV, run_executables

MAX_EXAMPLES = 5

VALUE_DTYPES = [
    pytest.param("bint8", id="bool-bint8", marks=pytest.mark.bool_values),
    pytest.param("int32", id="signed-int32", marks=pytest.mark.signed_integer),
    pytest.param("uint32", id="unsigned-uint32", marks=pytest.mark.unsigned_integer),
    pytest.param("float64", id="float64", marks=pytest.mark.floating),
    pytest.param(
        "complex[float64]",
        id="complex-float64",
        marks=pytest.mark.complex_values,
    ),
    pytest.param(
        "iso[int8]",
        id="iso-int8",
        marks=[pytest.mark.iso, pytest.mark.signed_integer],
    ),
]


def maybe_run_executables(data: st.DataObject, **kwargs: Any) -> None:
    missing = [env for env in EXECUTABLE_ENV.values() if os.environ.get(env) is None]
    if missing:
        pytest.skip(f"missing executable environment variables: {', '.join(missing)}")
    generated = data.draw(npy_inputs(**kwargs), label="generated")
    run_executables(generated)


@pytest.mark.parametrize("values_dtype", VALUE_DTYPES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_1_dim(data: st.DataObject, values_dtype: str) -> None:
    maybe_run_executables(data, n=1, values_dtypes=values_dtype)


@pytest.mark.parametrize("values_dtype", VALUE_DTYPES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_2_dim(data: st.DataObject, values_dtype: str) -> None:
    maybe_run_executables(data, n=2, values_dtypes=values_dtype)


@pytest.mark.parametrize("values_dtype", VALUE_DTYPES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_csr(data: st.DataObject, values_dtype: str) -> None:
    maybe_run_executables(data, format_name="CSR", values_dtypes=values_dtype)
