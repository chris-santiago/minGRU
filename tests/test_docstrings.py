"""Docstring-coverage tests for the public API.

Every symbol in ``mingru.__all__`` (bar the ``__version__`` string) must
carry a NumPy-style docstring, and the runnable examples in
``mingru.min_gru`` must all pass. This is the rot-guard for the docstring
pass in Tasks 2-3.

Sections
--------
1. Coverage -- every public symbol has a non-empty docstring
2. NumPy structure -- section headers are underlined with matching dashes
3. Doctests -- every runnable example in mingru.min_gru passes
"""

from __future__ import annotations

import doctest

import mingru
from mingru import min_gru

SEED = 42

# NumPy-docstring (numpydoc) section headers, per the numpydoc standard.
_NUMPY_SECTIONS = frozenset(
    {
        "Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Other Parameters",
        "Raises",
        "Warns",
        "Warnings",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Attributes",
        "Methods",
    }
)


def _public_symbols():
    """Public API objects that should be documented (excludes __version__)."""
    return [
        (name, getattr(mingru, name))
        for name in mingru.__all__
        if name != "__version__"
    ]


# ===========================================================================
# 1. Coverage
# ===========================================================================


class TestDocstringCoverage:
    def test_version_excluded_is_a_string(self):
        # __version__ is data, not a documentable object.
        assert isinstance(mingru.__version__, str)

    def test_every_public_symbol_has_a_docstring(self):
        missing = [name for name, obj in _public_symbols() if not (obj.__doc__ or "").strip()]
        assert not missing, f"symbols missing a docstring: {missing}"

    def test_docstrings_have_a_summary_line(self):
        for name, obj in _public_symbols():
            first = obj.__doc__.strip().splitlines()[0].strip()
            assert first, f"{name} has an empty summary line"


# ===========================================================================
# 2. NumPy structure
# ===========================================================================


class TestNumpyStructure:
    def test_section_underlines_match(self):
        """A recognized section header must be underlined with '-' of equal length.

        Catches the classic numpydoc break where a header's dash underline is
        the wrong length (numpydoc then silently drops the section).
        """
        for name, obj in _public_symbols():
            lines = obj.__doc__.expandtabs().splitlines()
            for i, raw in enumerate(lines[:-1]):
                header = raw.strip()
                if header in _NUMPY_SECTIONS:
                    underline = lines[i + 1].strip()
                    if underline and set(underline) == {"-"}:
                        assert len(underline) == len(header), (
                            f"{name}: '{header}' underline length "
                            f"{len(underline)} != {len(header)}"
                        )

    def test_classes_document_parameters(self):
        """Each public class/function exposing constructor args names them."""
        for name, obj in _public_symbols():
            doc = obj.__doc__
            # Every public class and scan function in this API takes inputs,
            # so a Parameters section is expected.
            assert "Parameters" in doc, f"{name} has no Parameters section"


# ===========================================================================
# 3. Doctests
# ===========================================================================


class TestDoctests:
    def test_min_gru_doctests_pass(self):
        results = doctest.testmod(min_gru, verbose=False)
        assert results.attempted > 0, "no doctests were collected"
        assert results.failed == 0, f"{results.failed} doctest(s) failed"
