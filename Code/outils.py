import numpy as np
import random
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, x1, x2, x3, x4, y1, y2, y3, y4):
        self.p1 = Point(x1,y1)
        self.p2 = Point(x2, y2)
        self.p3 = Point(x3, y3)
        self.p4 = Point(x4, y4)

def vol(r):
    dx = r.x_max - r.x_min
    dy = r.y_max - r.y_min

    return dx*dy

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    dy = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if (dx>=0) and (dy>=0):
        return dx*dy

def extremum(r):
    x_min = min(r.p1.x, r.p2.x, r.p3.x, r.p4.x)
    x_max = max(r.p1.x, r.p2.x, r.p3.x, r.p4.x)
    y_min = min(r.p1.y, r.p2.y, r.p3.y, r.p4.y)
    y_max = max(r.p1.y, r.p2.y, r.p3.y, r.p4.y)

    r_ex = namedtuple(x_min, y_min, x_max, y_max)

    return r_ex

r1 = 

r1_ex = extremum(r1)
r2_ex = extremum(r2)

iou = area(r1_ex, r2_ex) / (vol(r1) + vol(r2) - 2*area(r1_ex, r2_ex))