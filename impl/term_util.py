# SPDX-License-Identifier: MIT
BARS_V = '  ▏▎▍▌▋▊▉█'
BARS_H = ' ▁▂▃▄▅▆▇█'

def gauge(score, width):
    '''Terminal gauge'''
    bw = score * width
    s = ''
    for x in range(width):
        # segment x covers x..x+1
        if x >= bw:
            ch = ' '
        elif (x + 1) >= bw:
            ch = BARS_V[int((bw - x) * 9.0)]
        else:
            ch = BARS_V[-1]
        s += ch
    return s

def colorize(fg, bg, glyph):
    return f'\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m{glyph}\x1b[0m'
