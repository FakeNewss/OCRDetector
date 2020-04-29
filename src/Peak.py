# -*- coding: utf-8 -*-
# Python version:   3.7
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('peakIndex', int32),               # a simple scalar field
    ('startPos', int32),               # a simple scalar field
    ('endPos', int32),               # a simple scalar field
    ('width', int32),               # a simple scalar field
    ('width', int32),               # a simple scalar field
    ('aveHeight', float32),               # a simple scalar field
    ('leftK', float32),               # a simple scalar field
    ('rightK', float32),          # a simple scalar field
]

@jitclass(spec)
class Peak:
    def __init__(self, peakIndex, startPos, endPos, width, leftK, rightK):
        self.peakIndex = peakIndex
        self.startPos = startPos
        self.endPos = endPos
        self.width = width
        self.aveHeight = 0.0
        self.leftK = leftK
        self.rightK = rightK
    def __str__(self):
        return 'startPos : %d, endPos : %d, width : %d' % (self.startPos, self.endPos, self.width)