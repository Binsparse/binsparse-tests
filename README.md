# Binsparse Compliance Test Suite

This repository contains the compliance test suite for libraries that support the
[Binsparse file format](https://github.com/GraphBLAS/binsparse-specification).

## Quick start

Install the project with [Pixi](https://pixi.sh/):

```console
pixi install
```

## Configure the library under test

Libraries must provide three executables that convert sparse arrays between NumPy
and Binsparse representations.

### `npy_to_binsparse`

```text
npy_to_binsparse <tensor_in> <pattern_in> <fill_value_in> <header_in> <tensor_out>
```

| Argument | Description |
| --- | --- |
| `tensor_in` | `.npy` file containing the dense input tensor |
| `pattern_in` | `.npy` file indicating which values are stored explicitly |
| `fill_value_in` | `.npy` file containing the input tensor's fill value |
| `header_in` | Partial Binsparse header whose fields must appear in the output |
| `tensor_out` | Destination Binsparse file |

The executable must complete the partial header while preserving all fields supplied
in `header_in`.

### `binsparse_to_npy`

```text
binsparse_to_npy <tensor_in> <tensor_out> <pattern_out> <fill_value_out>
```

| Argument | Description |
| --- | --- |
| `tensor_in` | Input Binsparse file |
| `tensor_out` | Destination `.npy` file containing the dense tensor values |
| `pattern_out` | Destination `.npy` file indicating which values were stored explicitly |
| `fill_value_out` | Destination `.npy` file containing the tensor's fill value |

### `binsparse_to_binsparse`

```text
binsparse_to_binsparse <tensor_in> <tensor_out>
```

| Argument | Description |
| --- | --- |
| `tensor_in` | Input Binsparse file |
| `tensor_out` | Destination Binsparse file |

The library must convert the input to its internal representation before writing the
output.

If all three executables are in the same directory, set `BINSPARSE_BIN` to that
directory:

```console
export BINSPARSE_BIN=/path/to/bin
```

The directory must contain executables named `npy_to_binsparse`,
`binsparse_to_npy`, and `binsparse_to_binsparse`.

Alternatively, set each executable path individually:

```console
export NPY_TO_BINSPARSE=npy_to_binsparse
export BINSPARSE_TO_NPY=binsparse_to_npy
export BINSPARSE_TO_BINSPARSE=binsparse_to_binsparse
```

An individual executable variable takes precedence over `BINSPARSE_BIN` when both
are set.

### Select a Binsparse version

Set `BINSPARSE_TESTS_VERSION` to choose the specification version used by the tests:

```console
export BINSPARSE_TESTS_VERSION="2023.12"
```

## Run the suite

By default, the suite tests NPZ, Zarr, and HDF5:

```console
pixi run -e test test
```

Use pytest marks to test only selected container types:

```console
pixi run -e test-npz pytest -m npz
pixi run -e test-zarr pytest -m zarr
pixi run -e test-hdf5 pytest -m hdf5
pixi run -e test pytest -m "npz or hdf5"
```

## Definition of conformance

The suite tests whether a library preserves the semantic meaning of arrays stored in
Binsparse files. Two arrays are equivalent when:

1. Their dimensions, data types, and values match at every corresponding index.
2. They have the same explicit storage pattern. An explicitly stored fill value is
   semantically distinct from an implicit fill value.
3. They have the same fill value, even when neither array stores any values implicitly.

Storage format, index type, and whether values use an iso representation are not
semantically meaningful, but should be preserved through a roundtrip.

`npy_to_binsparse` tests whether the library can construct an equivalent Binsparse
tensor from its dense values, explicit storage pattern, fill value, and requested
header fields. `binsparse_to_binsparse` tests whether the library's internal
representation can represent tensors read from different Binsparse formats. Together,
these conversions test writing and round-tripping through the library's internal
state. The direct `binsparse_to_npy` tests provide clearer diagnostics when a
conversion fails.

### Test generation

The suite uses [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to generate
a diverse set of random matrices. It tests every predefined alias and a random set of
custom formats. Framework-relevant capabilities are exposed as pytest parameters:
test IDs distinguish zero from nonzero fill values; dense, flat COO, and mixed
custom layouts; and transposed from non-transposed custom tensors. This allows skip
and XFAIL files to select unsupported capabilities without suppressing every
Hypothesis example in a broader parameter group.

### Binary container equivalence

Binary containers can include metadata unrelated to array semantics. For example,
HDF5 records timestamps. The suite compares only relevant properties and ignores
details such as timestamps and JSON field ordering.

## Skip or mark expected failures

Use skip and XFAIL files to identify tests that should be skipped or expected to fail.
XFAIL tests still run and are reported as XPASS when they unexpectedly pass.

The default files are `skips.txt` and `fails.txt` in the repository root. Use
`--skips-file` and `--xfails-file` to select other files. Either option may be repeated;
entries from all supplied files are merged:

```console
pytest \
  --skips-file skips-general.txt \
  --skips-file skips-macos.txt \
  binsparse_tests/
```

Each file contains test ID substrings, one per line. Empty lines and lines beginning
with `#` are ignored.

```text
# Skip a specific test, such as a custom format that is not CSF.
binsparse_tests/test_custom.py::test_sparse_sparse_dense

# Skip a specific parameter, such as an unsupported iso representation.
binsparse_tests/test_add.py::test_add[iso]

# Skip an entire module, such as all custom-format tests.
binsparse_tests/test_custom.py
```
