"""
This module computes the version this package using setuptools package metadata.
"""
import sys

if sys.version_info <= (3, 8):
    from importlib_metadata import version
else:
    from importlib.metadata import version


__version__ = version("connectfour")
