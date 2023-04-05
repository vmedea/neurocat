# SPDX-License-Identifier: MIT
import colorsys

def hex_to_rgb(rgbhex):
    if rgbhex[0] == '#':
        rgbhex = rgbhex[1:]
    if len(rgbhex) != 6:
        raise ValueError
    return (int(rgbhex[0:2], 16), int(rgbhex[2:4], 16), int(rgbhex[4:6], 16))

def rgb_to_hex(rgb, prefix_hash=True):
    if prefix_hash:
        prefix = '#'
    else:
        prefix = ''
    for i in range(3):
        assert(rgb[i] >= 0 and rgb[i] <= 255)
    return f'{prefix}{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def rgbf_to_rgb8(color):
    return tuple(int(max(min(comp * 256.0, 255.0), 0.0)) for comp in color)

def hsv_to_rgb8(h, s, v):
    color = colorsys.hsv_to_rgb(h, s, v)
    return rgbf_to_rgb8(color)

def lerp(rgb1, rgb2, w):
    w = max(0.0, min(1.0, w))

    return [int(rgb1[i] + w * (rgb2[i] - rgb1[i])) for i in range(3)]

def normalize_color(col, strength=255):
    v = max(col)
    if v > 0:
        col = [col[i] * strength // v for i in range(3)]

    col = [max(min(col[i], 255), 0) for i in range(3)]
    return tuple(col)
