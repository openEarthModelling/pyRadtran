"""Module entry point: ``python -m pyradtran`` delegates to the config CLI."""

import sys

from pyradtran.config.cli import main

if __name__ == "__main__":
    sys.exit(main())
