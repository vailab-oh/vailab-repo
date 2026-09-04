"""Compatibility launcher for the former research-script filename.

The supported API lives in :mod:`ta_rrt`. Run ``python -m ta_rrt --help`` for the
command-line interface.
"""

from ta_rrt.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
