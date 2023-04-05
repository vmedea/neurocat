#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
'''
Generate array of solid color embeddings.
'''
import argparse
import colorsys
import pickle
import sys

import numpy as np
from PIL import Image

from impl import clip_model
from impl.color_util import hex_to_rgb, rgb_to_hex, hsv_to_rgb8
from impl import resource

def all_colors_rgb():
    # grey ramp
    colors = []
    for val in np.linspace(0.0, 1.0, 9):
        colors.append(hsv_to_rgb8(0.0, 0.0, val))
    # three brightness levels
    for val in np.linspace(0.0, 1.0, 4)[1:]:
        # three saturation levels
        for sat in np.linspace(0.0, 1.0, 4)[1:]:
            # 24 hues
            for hue in np.linspace(0.0, 1.0, 25)[:-1]:
                colors.append(hsv_to_rgb8(hue, sat, val))

    return colors


def parse_args():
    parser = argparse.ArgumentParser(description="Build neurocat color vectors.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Pytorch device to use. Default is 'cuda'. Use 'cpu' to not use GPU acceleration (slow)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    colors = all_colors_rgb()

    m = clip_model.VisionModel(args.device)

    # make embeddings for solid-colored squares using vision model
    embeddings = []
    for color in colors:
        image = Image.new('RGB', (224, 224), color=color)
        emb = m.embedding_from_image(image)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # store generated embeddings
    with resource.open('colors.npy', 'wb') as f:
        np.save(f, np.array(colors, dtype=np.uint8), allow_pickle=False)
        np.save(f, embeddings.astype(np.float16), allow_pickle=False)


if __name__ == '__main__':
    main()
