# Copyright (c) OpenMMLab. All rights reserved.
"""Suppress noisy warnings during training and evaluation."""
import sys
import warnings


# Lines containing any of these (substring) are not written to stderr.
STDERR_SUPPRESS_PATTERNS = [
    "[WARNING]your gpu arch",
    "isn't compiled in prebuilt",
    "may cause invalid device function",
    "available: {",  # spconv COMPILED_CUDA_ARCHS
]


class FilteredStderr:
    """Wrapper for stderr that drops lines matching known noisy patterns (e.g. spconv GPU arch)."""

    def __init__(self, real_stderr):
        self._real = real_stderr
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = s.decode("utf-8", errors="replace")
        self._buf += s
        while "\n" in self._buf or "\r" in self._buf:
            sep = "\n" if "\n" in self._buf else "\r"
            before, _, after = self._buf.partition(sep)
            line, self._buf = before, after
            if not self._should_suppress(line):
                self._real.write(line + sep)
        self._real.flush()

    def _should_suppress(self, line):
        return any(p in line for p in STDERR_SUPPRESS_PATTERNS)

    def flush(self):
        if self._buf and not self._should_suppress(self._buf):
            self._real.write(self._buf)
        self._buf = ""
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_suppress():
    """Install stderr filter and set warning filters. Call at start of main() in train/test."""
    # Suppress spconv GPU arch messages (they use print(..., file=sys.stderr))
    if not isinstance(sys.stderr, FilteredStderr):
        sys.stderr = FilteredStderr(sys.stderr)

    # Common noisy warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.distributed\.launch.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*deprecated.*will be removed.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Setting OMP_NUM_THREADS.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*MKL_NUM_THREADS.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*config is now expected to have.*runner.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*floor_divide is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*To copy construct from a tensor.*clone\(\)\.detach.*",
        category=UserWarning,
    )
