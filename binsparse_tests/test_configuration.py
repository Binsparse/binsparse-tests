from pathlib import Path

import pytest

from .run import EXECUTABLE_ENV, _resolve_commands

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def clear_executable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINSPARSE_BIN", raising=False)
    for variable in EXECUTABLE_ENV.values():
        monkeypatch.delenv(variable, raising=False)


def test_binsparse_bin_supplies_executable_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BINSPARSE_BIN", str(tmp_path))
    commands = _resolve_commands(
        npy_to_binsparse=None,
        binsparse_to_npy=None,
        binsparse_to_binsparse=None,
    )

    assert commands == {
        executable: str(tmp_path / executable) for executable in EXECUTABLE_ENV
    }


def test_individual_variable_overrides_binsparse_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BINSPARSE_BIN", str(tmp_path))
    monkeypatch.setenv("BINSPARSE_TO_NPY", "/custom/binsparse_to_npy")
    commands = _resolve_commands(
        npy_to_binsparse=None,
        binsparse_to_npy=None,
        binsparse_to_binsparse=None,
    )

    assert commands["binsparse_to_npy"] == "/custom/binsparse_to_npy"


def test_repeatable_skip_and_xfail_files(pytester: pytest.Pytester) -> None:
    pytester.makeconftest((Path(__file__).parents[1] / "conftest.py").read_text())
    pytester.makepyfile(
        test_options="""
        def test_pass():
            pass

        def test_skip_one():
            assert False

        def test_skip_two():
            assert False

        def test_expected_failure():
            assert False
        """
    )
    first_skips = pytester.makefile(".txt", first_skips="test_skip_one\n")
    second_skips = pytester.makefile(
        ".txt", second_skips="  # comment\n\ntest_skip_two  \n"
    )
    xfails = pytester.makefile(".txt", expected_failures="expected_failure\n")

    result = pytester.runpytest(
        "--skips-file",
        str(first_skips),
        "--skips-file",
        str(second_skips),
        "--xfails-file",
        str(xfails),
    )

    result.assert_outcomes(passed=1, skipped=2, xfailed=1)


def test_unmatched_pattern_warns(pytester: pytest.Pytester) -> None:
    pytester.makeconftest((Path(__file__).parents[1] / "conftest.py").read_text())
    pytester.makepyfile("def test_pass(): pass")
    skips = pytester.makefile(".txt", skips="missing_test\n")

    result = pytester.runpytest("--skips-file", str(skips))

    result.assert_outcomes(passed=1, warnings=1)
    result.stdout.fnmatch_lines(["*missing_test*"])


def test_default_skip_and_xfail_files(pytester: pytest.Pytester) -> None:
    pytester.makeconftest((Path(__file__).parents[1] / "conftest.py").read_text())
    pytester.makepyfile(
        """
        def test_skipped():
            assert False

        def test_expected_failure():
            assert False
        """
    )
    pytester.makefile(".txt", skips="test_skipped\n")
    pytester.makefile(".txt", fails="test_expected_failure\n")

    result = pytester.runpytest()

    result.assert_outcomes(skipped=1, xfailed=1)
