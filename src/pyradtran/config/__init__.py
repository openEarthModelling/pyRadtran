"""YAML configuration front-end for pyRadtran.

A config file describes one experiment (scene + aerosol + analysis intents);
:func:`~pyradtran.config.load_config` maps it onto the same builders the
Python API uses, so YAML and API-built scenes are interchangeable.
"""

from pyradtran.config.loader import (
    LoadedConfig,
    build_aerosol,
    build_scene,
    load_config,
)
from pyradtran.config.schema import CONFIG_VERSION, PyRadtranConfig

__all__ = [
    "CONFIG_VERSION",
    "LoadedConfig",
    "PyRadtranConfig",
    "build_aerosol",
    "build_scene",
    "load_config",
]
