# SPDX-License-Identifier: MIT
import pathlib

def filename(name):
    return pathlib.Path(__file__).parent.parent / 'data' / name

_builtin_open = open
def open(name, mode):
    return _builtin_open(filename(name), mode)
