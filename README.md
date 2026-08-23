# Test Suite for Binsparse Compliance

This is the test suite for libraries supporting the [Binsparse](https://github.com/GraphBLAS/binsparse-specification) file format.

# Quickstart

To run the tests, install the repo using pixi

```
pixi install .
```

## Specifying the parser to test

Libraries must implement sparse array format conversion in order to be tested. Three executables are required:

`npy_to_binsparse [tensor_in] [pattern_in] [fill_value_in] [header_in] [tensor_out]`

where `tensor_in` is an npy file holding the input tensor values in a dense format, and `pattern_in` is an npy file holding the description of which fill values are to be stored explicitly. `header_in` holds a partial binsparse header whose fields should be present in the output, the executable should complete the conversion.

`binsparse_to_npy [tensor_in] [tensor_out] [pattern_out] [fill_value_out]`

where `tensor_in` is a binsparse file holding the input tensor and `tensor_out` and `pattern_out` are npy files holding the output tensor values, whether each value was implicit, `fill_value_out` holds output fill value.

`binsparse_to_binsparse [tensor_in] [tensor_out]`

where `tensor_in` is a binsparse file holding the input tensor and `tensor_out` is a binsparse file holding the output. The library should convert to its internal representation in the middle.

The executables can be specified with the "NPY_TO_BINSPARSE", "BINSPARSE_TO_NPY", and "BINSPARSE_TO_BINSPARSE" environment variables, e.g.:

```
$ export NPY_TO_BINSPARSE=npy_to_binsparse
$ export BINSPARSE_TO_NPY=npy_to_binsparse
```

## Specifying the Binsparse Version

You can specify the Binsparse version to use when testing via the BINSPARSE_TESTS_VERSION environment variable, e.g.

```
$ export ARRAY_API_TESTS_VERSION="2023.12"
```

## Run the suite

You may run any of the following targets to test your executable against the appropriate binary container:

```
pixi test-hdf5
pixi test-zarr
pixi test-npz
```

# Our definition of conformance

We are interested in array libraries conforming to the binsparse spec, and that they understand the "semantic meaning" of the arrays contained in binsparse files. We define two arrays to be "the same" under the following conditions:
1. The value of the array at equivalent indices is the same (they are the same as dense arrays in dimension, value, and type)
2. We consider implicit versus explicit zeros to be semantically meaningful, so two arrays mean the same thing if they store the same values.
3. The arrays have the same "fill value," regardless of whether any elements are actually stored implicitly.


Thus, npy_to_binsparse tests whether the library understands how to convert the dense, pattern, and fill components of a tensor into an equivalent binsparse file. 

Array components like the storage format, whether values are iso, or the type of the indices, are not semantically meaningful. Still, binsparse_to_binsparse tests whether the libraries internal representation of a tensor distinguishes between different binsparse format representations.

Together, npy_to_binsparse and binsparse_to_binsparse are enough to fully test that a library conforms to binsparse, as it can read a semantic representation to internal state, then to a given binsparse file. It can also roundtrip through the same internal state, which is enough. However, we also provide tests for binsparse_to_npy to make errors clearer.

## Test generation

We use [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to generate a diverse set of random matrices. We test all predefined aliases, and a random set of custom formats.

## Binary container equivalence

Comparing binary containers is sometimes complicated. HDF5, for example, records a time stamp in the output. The test suite comes equipped with utilities to compare only the relevant properties of binary containers, and ignore things like the ordering of fields in json.

Skip or XFAIL test cases

Test cases you want to skip can be specified in a skips or XFAILS file. The difference between skip and XFAIL is that XFAIL tests are still run and reported as XPASS if they pass.

By default, the skips and xfails files are skips.txt and fails.txt in the root of this repository, but any file can be specified with the --skips-file and --xfails-file command line flags.

Both flags can be given several times, in which case the files are merged. This is useful to keep entries which only apply to some platforms separate from the general ones:

pytest --skips-file skips-general.txt --skips-file skips-macos.txt array_api_tests/

The files should list the test ids to be skipped/xfailed. Empty lines and lines starting with # are ignored. The test id can be any substring of the test ids to skip/xfail.

# skips.txt or xfails.txt
# Line comments can be denoted with the hash symbol (#)

# Skip specific test case, e.g. custom formats that are not CSF
binsparse_tests/test_custom.py::test_sparse_sparse_dense

# Skip specific test case parameter, e.g. you forgot to implement iso
binsparse_tests/test_add[iso]

# Skip module, e.g. no custom formats supported
binsparse_tests/test_custom.py

