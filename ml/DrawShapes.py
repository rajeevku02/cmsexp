import cv2

points = []

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

def clear():
    global points
    points = []