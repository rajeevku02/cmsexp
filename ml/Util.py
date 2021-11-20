import math

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

VERBOSE = False

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def dist(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx*dx + dy*dy)

def dist2(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def pt(p):
    return Point(p.x * CANVAS_WIDTH, p.y*CANVAS_HEIGHT, p.z*CANVAS_WIDTH)

def log(msg):
    if VERBOSE:
        print(msg)