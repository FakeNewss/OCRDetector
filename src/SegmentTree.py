# -*- coding: utf-8 -*-
# Python version:   3.7
#sourc https://www.luogu.org/blog/Doveqise/solution-p3373

from numba import jitclass          # import the decorator
from numba import int32, int64, float32    # import the types

spec = [
    ('n', int64),               # a simple scalar field
    ('num', int64[:]),          # an array field
    ('addtag', int64[:]),          # an array field
    ('multag', int64[:]),          # an array field
    ('data', int64[:]),          # an array field
    ('num', int64[:]),          # an array field
]

# @jitclass(spec)
class SegmentTree:
    def __init__(self,n, alist): #建树
        self.num = alist[:]
        self.addtag = [0] * 4 * len(self.num)
        self.multag = [1] * 4 * len(self.num)
        self.siz = [0] * 4 * len(self.num)
        self.data = [0] * 4 * len(self.num)
        self.n = len(self.num)
        self.build(1, 1, n)

    def ls(self, x): #左儿子
        return x << 1

    def rs(self, x): #右儿子
        return x << 1 | 1

    def push_up(self, x):
        self.data[x] = self.data[self.ls(x)] + self.data[self.rs(x)]

    def add(self, x, k): #加操作
        self.data[x] += k * self.siz[x]
        self.addtag[x] += k

    def mul(self, x, k): #乘操作
        self.data[x] *= k
        self.addtag[x] *= k
        self.multag[x] *= k


    def push_down(self, x): #下放标记
        if (self.multag[x] != 1):
            if (self.siz[self.siz[self.ls(x)]] != 0):
                self.mul(self.ls(x), self.multag[x])
            if (self.siz[self.siz[self.rs(x)]] != 0):
                self.mul(self.rs(x), self.multag[x])
            self.multag[x] = 1
        if (self.addtag[x] != 0):
            if (self.siz[self.siz[self.ls(x)]] != 0):
                self.add(self.ls(x), self.addtag[x])
            if (self.siz[self.siz[self.rs(x)]] != 0):
                self.add(self.rs(x), self.addtag[x])
            self.addtag[x] = 0

    def build(self, x, l, r): #建树
        if (l == r):
            self.siz[x] = 1
            self.data[x] = self.num[l - 1]
            return
        mid = (l + r) >> 1
        self.build(self.ls(x), l, mid)
        self.build(self.rs(x), mid + 1, r)
        self.push_up(x)
        self.siz[x] = self.siz[self.ls(x)] + self.siz[self.rs(x)]

    def addupdate(self, x, ql, qr, l, r, k):
        if (ql <= l) & (r <= qr):
            self.add(x, k)
            return
        self.push_down(x)
        mid = (l + r) >> 1
        if (ql <= mid):
            self.addupdate(self.ls(x), ql, qr, l, mid, k)
        if (qr > mid):
            self.addupdate(self.rs(x), ql, qr, mid + 1, r, k)
        self.push_up(x)

    def mulupdate(self, x, ql, qr, l, r, k):
        if (ql <= l) & (r <= qr):
            self.mul(x, k)
            return
        self.push_down(x)
        mid = (l + r) >> 1
        if (ql <= mid):
            self.mulupdate(self.ls(x), ql, qr, l, mid, k)
        if (qr > mid):
            self.mulupdate(self.rs(x), ql, qr, mid + 1, r, k)
        self.push_up(x)


    def query(self, x, ql, qr, l, r):
        res = 0
        if (ql <= l) & (r <= qr):
            return self.data[x]
        self.push_down(x)
        mid = (l + r) >> 1
        if (ql <= mid):
            res += self.query(self.ls(x), ql, qr, l, mid)
        if (qr > mid):
            res += self.query(self.rs(x), ql, qr, mid + 1, r)
        return res