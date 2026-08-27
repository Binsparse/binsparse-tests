from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("binsparse-tests")
    group.addoption(
        "--skips-file",
        action="append",
        default=None,
        metavar="FILE",
        help=(
            "file containing test ID substrings to skip; may be repeated to merge "
            "files (default: skips.txt)"
        ),
    )
    group.addoption(
        "--xfails-file",
        action="append",
        default=None,
        metavar="FILE",
        help=(
            "file containing test ID substrings to xfail; may be repeated to merge "
            "files (default: fails.txt)"
        ),
    )


def _load_patterns(
    filenames: Sequence[str] | None,
    default_name: str,
) -> dict[str, Path]:
    files = [Path(filename).expanduser() for filename in filenames or ()]
    if not files:
        default_file = Path(__file__).parent / default_name
        if default_file.exists():
            files = [default_file]

    patterns: dict[str, Path] = {}
    for file in files:
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                pattern = line.strip()
                if pattern and not pattern.startswith("#"):
                    patterns[pattern] = file
    return patterns


def _apply_patterns(
    items: list[pytest.Item],
    patterns: dict[str, Path],
    marker: pytest.MarkDecorator,
    option: str,
) -> None:
    matched = dict.fromkeys(patterns, False)
    for item in items:
        for pattern, source in patterns.items():
            if pattern in item.nodeid:
                item.add_marker(marker(reason=f"{option} ({source})"))
                matched[pattern] = True
                break

    unmatched = [
        pattern for pattern, was_matched in matched.items() if not was_matched
    ]
    if unmatched:
        entries = "\n".join(
            f"    {pattern} ({patterns[pattern]})" for pattern in unmatched
        )
        warnings.warn(
            pytest.PytestWarning(
                f"{len(unmatched)} pattern(s) from {option} files did not match any "
                f"collected tests:\n{entries}"
            ),
            stacklevel=1,
        )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    skip_patterns = _load_patterns(config.getoption("--skips-file"), "skips.txt")
    xfail_patterns = _load_patterns(config.getoption("--xfails-file"), "fails.txt")
    _apply_patterns(items, skip_patterns, pytest.mark.skip, "--skips-file")
    _apply_patterns(items, xfail_patterns, pytest.mark.xfail, "--xfails-file")
