import math

RANGE_DELTA = 0.5

def dist(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx*dx + dy*dy)

def dist2(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class LineEq:
    def __init__(self, p1, p2):
        self.A = p2.y - p1.y
        self.B = p1.x - p2.x
        self.C = p1.y * p2.x - p1.x * p2.y
    
    def dist_from(self, pt):
        dist = abs(self.A * pt.x + self.B * pt.y + self.C) / math.sqrt(self.A * self.A + self.B * self.B)
        return dist

    def solve(self, eq2):
        D = self.A * eq2.B - eq2.A * self.B
        Dx = eq2.B * (-self.C) - self.B * (-eq2.C)
        Dy = self.A *(-eq2.C) - eq2.A *(-self.C)
        if D == 0:
            return False, None
        return True, Point(Dx / D, Dy / D, 0)

    def angle_with_hz(self):
        return math.atan2(self.A, -self.B)

    def crosses(self, p1, p2):
        d1 = self.A * p1.x + self.B * p1.y + self.C
        d2 = self.A * p2.x + self.B * p2.y + self.C
        return d1 * d2 <= 0

    def perp(self, p):
        eq = LineEq(Point(0, 0, 0), Point(1, 1, 1))
        eq.A = -self.B
        eq.B = self.A
        eq.C = -eq.A * p.x - eq.B * p.y
        return eq

class LineSegment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def length(self):
        return dist(self.p1, self.p2)

    def eq(self):
        return LineEq(self.p1, self.p2)

    def contains(self, pt):
        d1 = dist(pt, self.p1)
        d2 = dist(pt, self.p2)
        len = self.length()
        diff = abs(len - (d1 + d2))
        return diff < RANGE_DELTA

    def dist_from(self, point):
        eq = LineEq(self.p1, self.p2)
        eq2 = eq.perp(point)
        _, near = eq.solve(eq2)
        if self.contains(near):
            return eq.dist_from(point)
        d1 = dist(self.p1, point)
        d2 = dist(self.p2, point)
        return min(d1, d2)

    def intersect(self, seg):
        eq1 = LineEq(self.p1, self.p2)
        eq2 = LineEq(seg.p1, seg.p2)
        got, at = eq1.solve(eq2)    
        if not got or not self.contains(at) or not seg.contains(at):
            return False, Point(0, 0, 0)
        return True, at

    def intersect_eq(self, eq):
        eq1 = LineEq(self.p1, self.p2)
        got, at = eq1.solve(eq)
        if not got or not self.contains(at):
            return False, Point(0, 0, 0)
        return True, at
