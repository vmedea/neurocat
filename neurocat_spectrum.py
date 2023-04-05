#!/usr/bin/env python3
# neural color spectrum
# SPDX-License-Identifier: MIT
import argparse
import sys

import numpy as np

from impl.color_assoc import ColorAssoc
from impl.clip_util import normalize
from impl.color_util import lerp, normalize_color, hsv_to_rgb8
from impl.term_util import gauge, BARS_H
from impl.word_db import WordDB

# XXX actual argument parsing.
# Visual settings.
chart_bg = (7, 7, 7)
spectro_bg = (7, 7, 7)
black_substitute = (24, 24, 24)

# Show all colors as white. This is best for judging relative values, but makes it more
# difficult to tell what is what.
# XXX add a legend/axis.
spectrum_bw = False

# Display bars instead of blocks.
#color_bars = False
color_bars = True

# Make sure all colors have the same brightness. As brightness is used as an axis, this
# makes it possible to distinguish relative values better.
# (use this with color_bars=False)
spectrum_norm = False

#subtract_abstract = False
subtract_abstract = True


def parse_args():
    parser = argparse.ArgumentParser(description="Print 'CLIP neural spectrum' for a word.")
    parser.add_argument(
        "word",
        type=str,
        help='The word to use',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wdb = WordDB()
    cas = ColorAssoc(subtract_abstract)

    m = wdb.lookup(args.word)

    if m is None:
        print(f'The word {args.word} is not in the database')
        sys.exit(1)

    scores, ac_score = cas.compute_scores(m)

    ########### Mapping of ranking and dot products to [0..1].

    def mapping_expdist(scores):
        '''
        negative exponential mapping (noisy)
        displays strongest associations
        '''
        if cas.subtract_abstract:
            steepness = 35.0
        else:
            steepness = 48.5

        scores_norm = np.exp(scores * steepness)
        scores_norm /= max(scores_norm)
        return np.clip(scores_norm, 0.0, 1.0)

    def mapping_expdist_norm(scores, ac_score):
        '''
        '''
        min_score = min(min(scores), ac_score)
        max_score = max(max(scores), ac_score)

        scores_norm = (scores - min_score) / (max_score - min_score)

        steepness = 4.0
        scores_norm = np.exp(scores_norm * steepness)
        scores_norm /= max(scores_norm)
        return np.clip(scores_norm, 0.0, 1.0)

    def mapping_polynomial(scores, ac_score, cutoff=0.50, power=1.6):
        '''
        negative exponential mapping (noisy)
        displays strongest associations
        '''
        # count ac_score in range determination
        # even though it's not displayed as an actual color
        min_score = min(min(scores), ac_score)
        max_score = max(max(scores), ac_score)

        if (max_score - min_score) < 0.01:
            # not possible to significantly distinguish anything
            return np.ones(len(scores))
        scores_norm = (scores - min_score) / (max_score - min_score)
        scores_norm -= cutoff # cut off lower half
        scores_norm = np.maximum(scores_norm, 0.0)
        scores_norm /= (1.0 - cutoff)

        scores_norm = scores_norm ** power
        return np.clip(scores_norm, 0.0, 1.0)

    def mapping_rank(scores):
        '''
        assigned based on simple rank
        this loses information and it would be good to incorporate somehow the distances
        after spending a long time trying to get the scaling right, this seems to work surprisngly well
        '''
        ranking = np.argsort(scores)
        scores_norm = np.zeros([len(ranking)])
        for i, r in enumerate(ranking):
            scores_norm[r] = np.exp(-(1.0 - i / len(ranking)) * 10.0)

        # clamp relative scores to 0..1
        return np.clip(scores_norm, 0.0, 1.0)

    def mapping_binary(scores, threshold=0.0):
        '''
        all colors with any positive association as full intensity
        doesn't show strength of association but shows the color better

        because of this it works well with spectrum_norm=False
        '''
        scores_norm = np.zeros(len(scores))
        for i in range(len(scores)):
            if scores[i] > threshold:
                scores_norm[i] = 1.0
        return scores_norm

    def print_statistics(scores):
        # Print some statistics
        min_score = min(scores)
        max_score = max(scores)
        median = np.percentile(scores, 50)
        print(f'median {median * 100.0:3.1f} range [{(min_score) * 100.0:5.1f}..{(max_score) * 100.0:5.1f}] max-min {(max_score - min_score) * 100.0:5.1f} max-median {(max_score-median) * 100:5.1f}')

        #bins, edges = np.histogram(scores)
        #print(list(bins))
        #print(list(edges * 100.0))

    print_statistics(scores)

    #scores_norm = mapping_rank(scores)
    #scores_norm = mapping_expdist(scores)
    scores_norm = mapping_expdist_norm(scores, ac_score)
    #scores_norm = mapping_polynomial(scores, ac_score)
    #scores_norm = mapping_binary(scores)

    #### Visualization

    def gen_greyramp(n, darkest=0, rep=1):
        for val in np.linspace(0.0, 1.0, n):
            col = tuple(max(comp, darkest) for comp in hsv_to_rgb8(0.0, 0.0, val))
            for _ in range(rep):
                yield col

    def gen_spectrum(n, sat, val):
        for hue in np.linspace(0.0, 1.0, n, endpoint=False):
            yield hsv_to_rgb8(hue, sat, val)

    class FixedGlyph:
        def __init__(self, fg, bg, glyph):
            self.fg = fg
            self.bg = bg
            self.glyph = glyph

    # value/brightness gradient
    blvl = [
        FixedGlyph((0x40, 0x40, 0x40), (0, 0, 0), '▕'),
        FixedGlyph((0x80, 0x80, 0x80), (0, 0, 0), '▕'),
        FixedGlyph((0xc0, 0xc0, 0xc0), (0, 0, 0), '▕'),
    ]

    hlvlc = [
        FixedGlyph(normalize_color(color, 50), (0, 0, 0), '▔') for color in gen_spectrum(24, 1.0, 1.0)
    ]
    hlvlg = [
        FixedGlyph(color, (0, 0, 0), '▔') for color in gen_greyramp(9, black_substitute[0], rep=3)
    ]

    # XXX move greyscale spectrum to the right to be aligned with the overall top-down bright-dark axis.
    color_rows = [
        (2, list(gen_greyramp(9, rep=3))),
        (1, list(gen_greyramp(9, rep=3))),
        (0, hlvlg),
        #(0, []),
        # brightest (three levels of saturation)
        (0, [blvl[2]] + list(gen_spectrum(24, 1.0 / 3.0, 3.0 / 3.0))),
        (0, [blvl[2]] + list(gen_spectrum(24, 2.0 / 3.0, 3.0 / 3.0))),
        (0, [blvl[2]] + list(gen_spectrum(24, 3.0 / 3.0, 3.0 / 3.0))),
        (0, []),
        # darker (three levels of saturation)
        (0, [blvl[1]] + list(gen_spectrum(24, 1.0 / 3.0, 2.0 / 3.0))),
        (0, [blvl[1]] + list(gen_spectrum(24, 2.0 / 3.0, 2.0 / 3.0))),
        (0, [blvl[1]] + list(gen_spectrum(24, 3.0 / 3.0, 2.0 / 3.0))),
        (0, []),
        # even darker
        (0, [blvl[0]] + list(gen_spectrum(24, 1.0 / 3.0, 1.0 / 3.0))),
        (0, [blvl[0]] + list(gen_spectrum(24, 2.0 / 3.0, 1.0 / 3.0))),
        (0, [blvl[0]] + list(gen_spectrum(24, 3.0 / 3.0, 1.0 / 3.0))),
        (0, [None] + hlvlc),
    ]

    for mode, row in color_rows:
        s = []
        for col in row:
            if col is None:
                s.append(' ')
                continue
            if isinstance(col, FixedGlyph):
                s.append(f'\x1b[38;2;{col.fg[0]};{col.fg[1]};{col.fg[2]};48;2;{col.bg[0]};{col.bg[1]};{col.bg[2]}m{col.glyph}\x1b[0m')
                continue
            idx = cas.rgbs.index(col)
            ii = min(1.0, max(0.0, scores_norm[idx]))

            if col == (0, 0, 0):
                col = black_substitute
            if mode == 2: # top row of graph
                fg = col
                bg = chart_bg
                glyph = BARS_H[max(min(int(ii * 17) - 8, 8), 0)]
            elif mode == 1: # bottom row of graph
                fg = col
                bg = chart_bg
                glyph = BARS_H[max(min(int(ii * 17), 8), 0)]
            else:
                # normalize color
                if spectrum_norm:
                    col = normalize_color(col, 255)

                if color_bars:
                    #fg = col
                    bg = chart_bg
                    fg = lerp(spectro_bg, col, ii)
                    glyph = BARS_H[min(int(ii * len(BARS_H)), len(BARS_H) - 1)]
                else:
                    #bg = lerp((0, 0, 0), col, ii)
                    #bg = (0, 0, 0)
                    fg = (0, 0, 0)
                    #fg = lerp(spectro_bg, (255, 255, 255), ii)
                    if spectrum_bw:
                        bg = lerp(spectro_bg, (255, 255, 255), ii)
                    else:
                        bg = lerp(spectro_bg, col, ii)
                    #glyph = '·' 
                    #glyph = '■'
                    #fg = (0, 0, 0)
                    #bg = col
                    glyph = ' '


            s.append(f'\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m{glyph}\x1b[0m')
        print(''.join(s))


if __name__ == '__main__':
    main()
