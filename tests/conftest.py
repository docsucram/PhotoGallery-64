"""Global test configuration and PySide6 process termination cleanup."""

import os
import pytest


def pytest_unconfigure(config):
    """
    On Windows with PySide6, standard C-runtime DLL unloading during process teardown
    can trigger 0xC0000005 access violations in COM/Qt plugin unloading after all tests
    have passed. Fast-exiting with the session exit status ensures clean exit code reporting.
    """
    if os.name == "nt":
        # Check exit status from session if available
        code = getattr(config, "_exitstatus", 0)
        os._exit(int(code))
