from geometry import dist, LineSegment, LineEq

active = False

THRESHOLD_1 = 0.3
THRESHOLD_2 = 0.4

def deactivate_drag1():
    global active
    active = False

def check_drag_1(pts):
    global active
    d1 = dist(pts[0], pts[5])
    d2 = dist(pts[4], pts[8])
    factor = 100.0
    if d1 != 0:
        factor = d2 / d1
    if active:
        if factor > THRESHOLD_2:
            active = False
    else:
        if factor < THRESHOLD_1:
            active = True
    return active