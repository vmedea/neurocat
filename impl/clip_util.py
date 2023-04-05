# SPDX-License-Identifier: MIT
import sys

import numpy

def normalize(v):
    '''Normalize a vector (L2 norm).'''
    return v / max(numpy.sqrt(numpy.sum(v * v)), 1e-8)

def l2norm(v):
    return numpy.sqrt(numpy.sum(v * v))

def cos_sim(a, b):
    '''Normalized dot product measure (if both arguments are already normalized, just use numpy.dot).'''
    return numpy.dot(a, b) / (norm(a) * norm(b))

