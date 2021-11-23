import cv2
from geometry import dist

points = []
DEL_DIST = 4

def draw_points(image, pts):
    if len(pts) < 2:
        return
    for i in range(0, len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        cv2.line(image, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 0), thickness=4)

def draw_shapes(image, tmp):
    for pts in points:
        draw_points(image, pts)
    draw_points(image, tmp)

def add_points(pts):
    points.append(pts)

def is_close(pts, pts2):
    for p in pts:
        for p2 in pts2:
            d = dist(p, p2)
            if d < DEL_DIST:
                return True
    return False

def del_shapes(pts):
    for i in reversed(range(0, len(points))):
        spts = points[i]
        if is_close(spts, pts):
            del points[i]

def move_shapes(pts):
    if len(pts) < 2:
        return
    dx = pts[-1].x - pts[0].x
    dy = pts[-1].y - pts[0].y
    for pts in points:
        for p in pts:
            p.x += dx
            p.y += dy

def undo():
    global points
    if len(points) > 0:
        del points[-1]

def clear():
    global points
    points = []