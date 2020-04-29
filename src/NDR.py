# -*- coding: utf-8 -*-
# Python version:   3.7
class NDR:
    def __init__(self, startPos, endPos, width):
        self.startPos = startPos
        self.endPos = endPos
        self.width = width
        self.aveHeight = 0
    def __str__(self):
        return 'startPos : %d, endPos : %d, width : %d' %(self.startPos, self.endPos, self.width)