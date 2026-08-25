import itertools
from copy import deepcopy

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

VERSION = "0.1"
MISSING = object()


def shapes(n, max_size=4):
    return st.tuples(*[st.integers(min_value=0, max_value=max_size) for _ in range(n)])


def formats(n):
    if n == 0:
        return st.just({"level_desc": "element"})
    return st.one_of(
        [
            st.fixed_dictionaries(
                {
                    "level_desc": st.sampled_from(["sparse", "dense"]),
                    "rank": st.just(r),
                    "level": formats(n - r),
                }
            )
            for r in range(1, n + 1)
        ]
    )


def dense(level, rank=1):
    return {"level_desc": "dense", "rank": rank, "level": deepcopy(level)}


def sparse(level, rank=1):
    return {"level_desc": "sparse", "rank": rank, "level": deepcopy(level)}


element = {"level_desc": "element"}

predefined = {
    "DVEC": (dense(element), None),
    "DMAT": (dense(dense(element)), None),
    "DMATR": (dense(dense(element)), None),
    "DMATC": (dense(dense(element)), (1, 0)),
    "CVEC": (sparse(element), None),
    "CSR": (dense(sparse(element)), None),
    "CSC": (dense(sparse(element)), (1, 0)),
    "DCSR": (sparse(sparse(element)), None),
    "DCSC": (sparse(sparse(element)), (1, 0)),
    "COO": (sparse(element, rank=2), None),
    "COOR": (sparse(element, rank=2), None),
    "COOC": (sparse(element, rank=2), (1, 0)),
}

predefined_by_rank = {
    1: ["DVEC", "CVEC"],
    2: ["DMATR", "DMATC", "CSR", "CSC", "DCSR", "DCSC", "COOR", "COOC"],
}

index_dtype_names = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

base_value_dtype_names = [
    "bint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]

complex_value_dtype_names = [
    *base_value_dtype_names,
    "complex[float32]",
    "complex[float64]",
]

dtype_to_str = {
    np.dtype("bool"): "bint8",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("complex64"): "complex[float32]",
    np.dtype("complex128"): "complex[float64]",
}

str_to_dtype = {value: key for key, value in dtype_to_str.items()}


def easy_index_datatypes():
    return st.sampled_from(["int32", "int64"])


def index_datatypes():
    return st.sampled_from(index_dtype_names)


def base_value_datatypes():
    return st.sampled_from(base_value_dtype_names)


def complex_value_datatypes():
    return st.sampled_from(complex_value_dtype_names)


def value_datatypes(allow_iso=True):
    base = complex_value_datatypes()
    if not allow_iso:
        return base
    return st.one_of(base, base_value_datatypes().map(lambda dtype: f"iso[{dtype}]"))


def trees(format, shape):
    match format["level_desc"]:
        case "dense":
            rank = format["rank"]
            subfibers = trees(format["level"], shape[rank:])
            keys = itertools.product(*[range(s) for s in shape[:rank]])
            return st.fixed_dictionaries(dict.fromkeys(keys, subfibers))
        case "sparse":
            rank = format["rank"]
            subfibers = trees(format["level"], shape[rank:])
            keys = list(itertools.product(*[range(s) for s in shape[:rank]]))
            if not keys:
                return st.just({})
            return st.dictionaries(st.sampled_from(keys), subfibers)
        case "element":
            return st.just(None)
        case _:
            raise ValueError("unrecognized level_desc")


def tree_coordinates(tree):
    coordinates = []

    def traverse(node, coord):
        if node is None:
            coordinates.append(coord)
        else:
            for key, value in node.items():
                traverse(value, (*coord, *key))

    traverse(tree, ())
    return sorted(coordinates)


def coordinates(format, shape):
    return trees(format, shape).map(tree_coordinates)


def pattern_from_coordinates(shape, coordinates):
    pattern = np.zeros(shape, dtype=bool)
    for coord in coordinates:
        pattern[coord] = True
    return pattern


@st.composite
def patterns(draw, format, shape):
    coords = draw(coordinates(format, shape))
    return pattern_from_coordinates(shape, coords)


def transposes(n):
    return st.permutations(range(n)).map(tuple)


def optional_transposes(n):
    return st.one_of(st.none(), transposes(n))


def dtypes(N, format, index_dtypes=None, pos_dtypes=None, values_dtypes=None):
    if rank(format) != N:
        raise ValueError("format rank does not match N")
    if index_dtypes is None:
        index_dtypes = index_datatypes()
    if pos_dtypes is None:
        pos_dtypes = easy_index_datatypes()
    if values_dtypes is None:
        values_dtypes = complex_value_datatypes()
    datatypes = {}

    def walk(level, depth, root):
        match level["level_desc"]:
            case "sparse":
                rank = level["rank"]
                if not root:
                    datatypes[f"pointers_to_{depth}"] = as_strategy(pos_dtypes)
                for i in range(depth, depth + rank):
                    datatypes[f"indices_{i}"] = as_strategy(index_dtypes)
                walk(level["level"], depth + rank, False)
            case "dense":
                walk(level["level"], depth + level["rank"], False)
            case "element":
                datatypes["values"] = as_strategy(values_dtypes)
            case _:
                raise ValueError("unrecognized level_desc")

    walk(format, 0, True)
    return st.fixed_dictionaries(datatypes)


def static_dtypes(format, values_dtype, index_dtype="int64", pos_dtype="int64"):
    datatypes = {}

    def walk(level, depth, root):
        match level["level_desc"]:
            case "sparse":
                rank = level["rank"]
                if not root:
                    datatypes[f"pointers_to_{depth}"] = pos_dtype
                for i in range(depth, depth + rank):
                    datatypes[f"indices_{i}"] = index_dtype
                walk(level["level"], depth + rank, False)
            case "dense":
                walk(level["level"], depth + level["rank"], False)
            case "element":
                datatypes["values"] = values_dtype
            case _:
                raise ValueError("unrecognized level_desc")

    walk(format, 0, True)
    return datatypes


def header(
    format,
    shape,
    pattern,
    values_dtype,
    format_name="custom",
    index_dtype="int64",
    pos_dtype="int64",
    transpose=None,
    fill=True,
):
    if rank(format) != len(shape):
        raise ValueError("format rank does not match shape rank")
    if transpose is not None and sorted(transpose) != list(range(len(shape))):
        raise ValueError("transpose must be a permutation of shape dimensions")
    if fill not in (None, False, True):
        raise ValueError("fill must be None or a boolean")
    data_types = static_dtypes(format, values_dtype, index_dtype, pos_dtype)
    if fill is True:
        data_types["fill_value"] = unwrapped_dtype(values_dtype)
    result = {
        "version": VERSION,
        "format": format_name,
        "shape": list(shape),
        "number_of_stored_values": int(np.sum(pattern)),
        "data_types": data_types,
    }
    if fill is not None:
        result["fill"] = fill
    if format_name == "custom":
        result["custom"] = {"level": deepcopy(format)}
        if transpose is not None:
            result["custom"]["transpose"] = list(transpose)
    return result


@st.composite
def headers(
    draw,
    format,
    shape,
    pattern,
    format_name="custom",
    index_dtypes=None,
    pos_dtypes=None,
    values_dtypes=None,
    transpose=MISSING,
    fill=MISSING,
):
    if index_dtypes is None:
        index_dtypes = easy_index_datatypes()
    if pos_dtypes is None:
        pos_dtypes = easy_index_datatypes()
    if values_dtypes is None:
        values_dtypes = value_datatypes(allow_iso=bool(np.any(pattern)))
    if transpose is MISSING:
        transpose = draw(optional_transposes(len(shape)))
    else:
        transpose = draw(as_strategy(transpose))
    if fill is MISSING:
        fill = draw(st.sampled_from([None, True]))
    else:
        fill = draw(as_strategy(fill))

    values_dtype = draw(as_strategy(values_dtypes))
    return header(
        format,
        shape,
        pattern,
        values_dtype,
        format_name,
        draw(as_strategy(index_dtypes)),
        draw(as_strategy(pos_dtypes)),
        transpose,
        fill,
    )


def predefined_header(
    format_name,
    shape,
    pattern,
    values_dtype,
    index_dtype="int64",
    pos_dtype="int64",
    transpose=None,
    fill=True,
):
    format, _ = predefined[format_name]
    return header(
        format,
        shape,
        pattern,
        values_dtype,
        format_name,
        index_dtype,
        pos_dtype,
        transpose,
        fill,
    )


def predefined_headers(format_name, shape, pattern, **kwargs):
    format, _ = predefined[format_name]
    return headers(format, shape, pattern, format_name, **kwargs)


@st.composite
def npy_inputs(
    draw,
    n=None,
    max_rank=3,
    max_size=4,
    predefined_only=False,
    format_name=None,
    values_dtypes=None,
    index_dtypes=None,
    pos_dtypes=None,
    transpose=MISSING,
    fill=MISSING,
):
    if format_name is not None:
        format, default_transpose = predefined[format_name]
        format_rank = rank(format)
        if n is not None and n != format_rank:
            raise ValueError(f"{format_name} has rank {format_rank}, not {n}")
        n = format_rank
    if n is None and predefined_only:
        n = draw(st.sampled_from(sorted(predefined_by_rank)))
    elif n is None:
        n = draw(st.integers(min_value=1, max_value=max_rank))
    elif predefined_only and n not in predefined_by_rank:
        raise ValueError(f"no predefined formats have rank {n}")

    shape = draw(shapes(n, max_size=max_size))

    if format_name is not None:
        if transpose is MISSING:
            transpose = default_transpose
        else:
            transpose = draw(as_strategy(transpose))
    elif predefined_only:
        format_name = draw(st.sampled_from(predefined_by_rank[n]))
        format, transpose = predefined[format_name]
    else:
        format_name = None
        format = draw(formats(n))
        if transpose is MISSING:
            transpose = draw(optional_transposes(n))
        else:
            transpose = draw(as_strategy(transpose))

    stored_shape = shape if transpose is None else tuple(shape[i] for i in transpose)
    stored_pattern = draw(patterns(format, stored_shape))
    pattern = (
        stored_pattern
        if transpose is None
        else np.transpose(stored_pattern, np.argsort(transpose))
    )

    if values_dtypes is None:
        values_dtype = draw(value_datatypes(allow_iso=bool(np.any(pattern))))
    else:
        values_dtype = draw(as_strategy(values_dtypes))
        if values_dtype.startswith("iso["):
            assume(bool(np.any(pattern)))
    if fill is MISSING:
        fill = draw(st.sampled_from([None, True]))
    else:
        fill = draw(as_strategy(fill))
    fill_value = draw(scalar_values(unwrapped_dtype(values_dtype)))
    stored_values = draw(value_array(values_dtype, int(np.sum(pattern))))
    tensor = dense_from_pattern(shape, pattern, fill_value, stored_values)
    fill_value = np.asarray(
        fill_value,
        dtype=str_to_dtype[unwrapped_dtype(values_dtype)],
    )
    index_dtype = draw(as_strategy("int64" if index_dtypes is None else index_dtypes))
    pos_dtype = draw(as_strategy("int64" if pos_dtypes is None else pos_dtypes))

    if format_name is None:
        out_header = header(
            format,
            shape,
            pattern,
            values_dtype,
            index_dtype=index_dtype,
            pos_dtype=pos_dtype,
            transpose=transpose,
            fill=fill,
        )
    else:
        out_header = predefined_header(
            format_name,
            shape,
            pattern,
            values_dtype,
            index_dtype=index_dtype,
            pos_dtype=pos_dtype,
            fill=fill,
        )
    return tensor, pattern, fill_value, out_header


def scalar_values(dtype):
    dtype = unwrapped_dtype(dtype)
    if dtype == "bint8":
        return st.booleans().map(np.bool_)
    if dtype.startswith("int"):
        np_dtype = np.dtype(dtype)
        info = np.iinfo(np_dtype)
        return st.integers(max(info.min, -1000), min(info.max, 1000)).map(np_dtype.type)
    if dtype.startswith("uint"):
        np_dtype = np.dtype(dtype)
        info = np.iinfo(np_dtype)
        return st.integers(0, min(info.max, 1000)).map(np_dtype.type)
    if dtype == "float32":
        return st.floats(
            width=32,
            allow_nan=False,
            allow_infinity=False,
        ).map(np.float32)
    if dtype == "float64":
        return st.floats(
            width=64,
            allow_nan=False,
            allow_infinity=False,
        ).map(np.float64)
    if dtype == "complex[float32]":
        return st.builds(
            lambda real, imag: np.complex64(complex(real, imag)),
            st.floats(width=32, allow_nan=False, allow_infinity=False),
            st.floats(width=32, allow_nan=False, allow_infinity=False),
        )
    if dtype == "complex[float64]":
        return st.builds(
            lambda real, imag: np.complex128(complex(real, imag)),
            st.floats(width=64, allow_nan=False, allow_infinity=False),
            st.floats(width=64, allow_nan=False, allow_infinity=False),
        )
    raise ValueError(f"unrecognized dtype {dtype!r}")


def value_array(dtype, size):
    np_dtype = str_to_dtype[unwrapped_dtype(dtype)]
    if dtype.startswith("iso["):
        return scalar_values(dtype).map(
            lambda value: np.full(size, value, dtype=np_dtype)
        )
    return st.lists(scalar_values(dtype), min_size=size, max_size=size).map(
        lambda values: np.asarray(values, dtype=np_dtype)
    )


def dense_from_pattern(shape, pattern, fill_value, values):
    if len(values) != int(np.sum(pattern)):
        raise ValueError("values must match the number of true pattern entries")
    dtype = np.result_type(np.asarray(fill_value).dtype, values.dtype)
    tensor = np.empty(shape, dtype=dtype)
    tensor[~pattern] = fill_value
    for coord, value in zip(np.argwhere(pattern), values, strict=True):
        tensor[tuple(coord)] = value
    return tensor


def unwrapped_dtype(dtype):
    if dtype.startswith("iso[") and dtype.endswith("]"):
        return dtype[4:-1]
    return dtype


def rank(format):
    if format["level_desc"] == "element":
        return 0
    return format["rank"] + rank(format["level"])


def as_strategy(value):
    if isinstance(value, st.SearchStrategy):
        return value
    return st.just(value)
