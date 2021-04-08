# -*- coding: utf-8 -*-
# Python version:   3.7
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