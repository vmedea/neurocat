# SPDX-License-Identifier: MIT
import numpy

from .term_util import colorize

def _do_boost_dark(fg):
    if fg == (0, 0, 0): # change black to some dark grey color
        fg = (80, 80, 80)
    elif sum(fg) < 0xa8:
        # very dark color, boost it a bit
        fg = (fg[0] * 2, fg[1] * 2, fg[2] * 2)
        fg = [min(x, 255) for x in fg]
    return fg

def fun_color(cas, wdb, line, m=None, highlight_unknown=True, boost_dark=True, fallback=False, multicolor=True, min_colorfulness=None):
    '''
    Outrageous word-coloring algorithm.
    If `multicolor` is set, color each letter, otherwise color only the entire word.
    '''
    if m is None:
        m = wdb.lookup(line)

    # fall back to remove 's' works sometimes (plurals)
    if m is None and fallback:
        if line.endswith('ic'):
            m = wdb.lookup(line[0:-2])
        elif line.endswith('es'):
            m = wdb.lookup(line[0:-2])
            if m is None:
                m = wdb.lookup(line[0:-1])
        elif line.endswith('s'):
            m = wdb.lookup(line[0:-1])

    if m is None:
        if highlight_unknown:
            return colorize((255, 255, 0), (255, 0, 0), line)
        else:
            return line
    else:
        scores, ac_score = cas.compute_scores(m)

        if min_colorfulness is not None:
            min_score = min(scores)
            max_score = max(scores)
            colorfulness = 1.0 - (ac_score - min_score) / (max_score - min_score)
            if colorfulness < min_colorfulness:
                return line

        if multicolor:
            indices = numpy.argsort(scores)
            indices = reversed(indices)
            s = []
            for glyph, idx in zip(line, indices):
                fg = cas.rgbs[idx]
                bg = (0, 0, 0)
                if boost_dark:
                    fg = _do_boost_dark(fg)
                s.append(colorize(fg, bg, glyph))
            return ''.join(s)
        else:
            idx = numpy.argmax(scores)
            fg = cas.rgbs[idx]
            if boost_dark:
                fg = _do_boost_dark(fg)
            bg = (0, 0, 0)
            return colorize(fg, bg, line)

