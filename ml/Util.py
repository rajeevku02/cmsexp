import math

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

from geometry import Point

VERBOSE = False

def pt(p):
    return Point(p.x * CANVAS_WIDTH, p.y*CANVAS_HEIGHT, p.z*CANVAS_WIDTH)

def log(msg):
    if VERBOSE:
        print(msg)