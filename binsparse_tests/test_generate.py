import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from .generate import datatypes, dense, element, npy_inputs


@pytest.mark.parametrize("fill_value_kind", ["zero", "nonzero"])
@given(data=st.data())
def test_fill_value_kind_is_enforced(
    data: st.DataObject,
    fill_value_kind: str,
) -> None:
    _, _, fill_value, _ = data.draw(
        npy_inputs(
            shape=st.just((2,)),
            format=st.just(dense(element)),
            datatypes=datatypes(st.just("int16")),
            fill_value_kind=fill_value_kind,
        )
    )

    assert fill_value.dtype == np.dtype("int16")
    assert bool(fill_value == 0) is (fill_value_kind == "zero")


@given(data=st.data())
def test_unknown_fill_value_kind_is_rejected(data: st.DataObject) -> None:
    with pytest.raises(ValueError, match="unknown fill value kind"):
        data.draw(
            npy_inputs(
                shape=st.just((1,)),
                format=st.just(dense(element)),
                datatypes=datatypes(st.just("int16")),
                fill_value_kind="missing",
            )
        )
