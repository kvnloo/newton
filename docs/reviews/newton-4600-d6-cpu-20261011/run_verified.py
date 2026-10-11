"""Assert the caller-selected Newton import before running a script or unittest."""
import pathlib
import runpy
import sys

import newton

checkout = pathlib.Path(sys.argv[1]).resolve()
expected = checkout / "newton" / "__init__.py"
assert pathlib.Path(newton.__file__).resolve() == expected, "Unexpected Newton import"
mode = sys.argv[2]
sys.argv = sys.argv[3:]
if mode == "script":
    runpy.run_path(sys.argv[0], run_name="__main__")
elif mode == "unittest":
    sys.argv.insert(0, "unittest")
    runpy.run_module("unittest", run_name="__main__")
else:
    raise ValueError("Expected script or unittest")
