from hypothesis import strategies as st

def shapes(n):
    return st.tuples(*[st.integers(min_size=0) for _ in range(n)])

def formats(n):
    if n == 0:
        return st.just({
            "level_desc": "element"
        })
    else:
        return st.one_of([
            st.fixed_dictionaries({
                "level_desc":st.sample_from(["sparse", "dense"]),
                "rank":r,
                "level":formats(n-r)
            })
            for r in range(n)
        ])

dtype_to_str = {
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
    np.dtype("complex32"): "complex[float32]",
    np.dtype("complex64"): "complex[float64]",
    np.dtype("bool"): "bint8",
}

str_to_dtype = {value: key for key, value in dtype_to_str.items()}

def easy_index_datatypes(n):
    return st.sample_from([
        "int32",
        "int64",
    ])

def index_datatypes(n):
    return st.sample_from([
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ])

def base_value_datatypes(n):
    st.sample_from([
        "bint8"
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
    ])

def complex_value_datatypes(n):
    st.sample_from([
        "bint8"
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
        "complex[float32]",
        "complex[float64]",
    ])

def patterns(format, shape):
    match format["level_desc"]:
        case "dense":
            rank = format["rank"]
            subfibers = patterns(format["level"], shape[rank:])
            return st.fixed_dictionaries(
                {key:subfibers for key in itertools.product(shape[:rank])}
                )
        case "sparse":
            rank = format["rank"]
            subfibers = patterns(format["level"], shape[rank:])
            return st.dictionaries(
                st.tuples(*[st.integers(min_size=0, max_size=s) for s in shape[:rank]]),
                subfibers
            )
        case _:
            raise ArgumentError("unrecognized")

def dtypes(N, format, index_dtypes, pos_dtypes, values_dtypes):
    level = format
    n = N
    datatypes = {}
    while n != 0:
        match level["level_desc"]:
            case "sparse":
                r = level["rank"]
                datatypes[f"pointers_to_{r}"] = pos_dtypes
                for i in range(n:n-r):
                    datatypes[f"indices_{i}"] = index_dtypes[i]
                n -= r
            case "dense":
                r = level["rank"]
                n -= r
                pass
    datatypes["values"] = values_dtypes
    return st.fixed_dictionaries(datatypes)