#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import sys

from impl import clip_model
from impl import word_db

def words(filename):
    words = set()
    if filename == '-':
        f = sys.stdin
    else:
        f = open(filename, 'r')
    with f:
        for line in f:
            word = line.strip()
            if word.startswith('#'): # skip comments
                continue
            if "'" in word or "-" in word or '.' in word or ' ' in word: # not single words, of very little use
                continue
            yield word


def parse_args():
    parser = argparse.ArgumentParser(description="Build neurocat word database.")
    parser.add_argument(
        "filename",
        nargs='?',
        default='/usr/share/dict/words',
        help="Name of words file to import, or '-' for standard input (default is /usr/share/dict/words)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Pytorch device to use. Default is 'cuda'. Use 'cpu' to not use GPU acceleration (slow)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = clip_model.TextModel(args.device)
    db = word_db.WordDB()

    for word in words(args.filename):
        if db.lookup(word) is None: # don't bother computing the embedding if word already in
            print(word)
            emb = model.embedding_from_text(word)
            db.insert(word, emb)

if __name__ == '__main__':
    main()
