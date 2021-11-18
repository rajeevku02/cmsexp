import math

# (720, 1280, 3)
MIN_DIST = 7.0
CACHE_SZ = 4

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def dist(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx*dx + dy*dy)

def pt(p):
    return Point(p.x * 1280, p.y*720, p.z*1280)

class LandmarkHandler:
    def __init__(self):
        self.pts1 = []
        self.pts2 = []

    def handle(self, landmarks):
        self.check(pt(landmarks[4]), pt(landmarks[8]))

    def check(self, p1, p2):
        if len(self.pts1) == 0:
            self.pts1.append(p1)
            self.pts2.append(p2)
            return
        d1 = dist(self.pts1[-1], p1)
        d2 = dist(self.pts2[-1], p2)
        if d1 < MIN_DIST and d2 < MIN_DIST:
            return
        self.pts1.append(p1)
        self.pts2.append(p2)
        if len(self.pts1) > CACHE_SZ:
            del self.pts1[0]
        if len(self.pts2) > CACHE_SZ:
            del self.pts2[0]
        print("d dx:", (p1.x - self.pts1[-2].x) - (p2.x - self.pts2[-2].x))
        print("d dy:", (p1.y - self.pts1[-2].y) - (p2.y - self.pts2[-2].y))
        