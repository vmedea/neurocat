#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import sys
import re

from impl.fun_color import fun_color
from impl.word_db import WordDB
from impl.color_assoc import ColorAssoc


def parse_args():
    parser = argparse.ArgumentParser(description="Show a file in CLIP neural association colors.")
    parser.add_argument(
        "filename",
        nargs='?',
        default='-',
        help="Name of file to process. The default is '-' (standard input)."
    )
    parser.add_argument(
        "-m", "--multicolor",
        type=int,
        default=1,
        help='Multiple colors per word instead of onlye one color per word. Value is 0 or 1. Default is 1.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cas = ColorAssoc()
    wdb = WordDB()

    filter_common = False
    min_colorfulness = None

    common_words = set()
    #with open('data/common_words.txt', 'r') as f:
    #    common_words = set(w.lower() for w in f.read().splitlines())
    #with open('data/funcolor_ignored_words.txt', 'r') as f:
    #    common_words.update((w.lower() for w in f.read().splitlines()))
    #with open('data/funcolor_add_words.txt', 'r') as f:
    #    common_words.difference_update((w.lower() for w in f.read().splitlines()))

    if args.filename != '-':
        f = open(args.filename, 'r')
    else:
        f = sys.stdin

    with f:
        for line in f:
            line = line[0:-1]
            words = re.split(r'(\W+)', line)
            for idx in range(0, len(words), 2):
                if not filter_common or (words[idx] not in common_words and len(words[idx]) > 3):
                    words[idx] = fun_color(cas, wdb, words[idx], highlight_unknown=False, fallback=True, multicolor=args.multicolor, min_colorfulness=min_colorfulness)
            print(''.join(words))

if __name__ == '__main__':
    main()
