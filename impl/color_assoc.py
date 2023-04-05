# SPDX-License-Identifier: MIT
import numpy as np

from .clip_util import normalize
from . import resource

class ColorAssoc:
    def __init__(self, subtract_abstract=True):
        self.subtract_abstract = subtract_abstract

        with resource.open('colors.npy', 'rb') as f:
            self.rgbs = np.load(f, allow_pickle=False)
            self.v = np.load(f, allow_pickle=False).astype(np.float32)

        self.rgbs = [tuple(col) for col in self.rgbs]
        # compute abstract color (midpoint between our colors)
        #self.abstract_color = normalize(sum(self.v))
        # compute abstract color (vector that results in the flattest spectrum "no color info")
        m = np.linalg.lstsq(self.v, np.ones(len(self.v)), rcond=None)[0]
        self.abstract_color = normalize(m)
        if subtract_abstract:
            for idx in range(len(self.v)):
                self.v[idx] = normalize(self.v[idx] - self.abstract_color)
            self.abstract_color = np.zeros(m.shape)

    def compute_scores(self, m):
        scores = np.matmul(self.v, m)
        ac_score = np.dot(self.abstract_color, m)
        return scores, ac_score

    def lookup_color(self, rgb):
        for i, c in enumerate(self.rgbs):
            if c == rgb:
                return self.v[i]
        raise KeyError
