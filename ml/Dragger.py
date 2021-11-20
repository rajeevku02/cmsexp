from Util import dist, dist2

DELTA = 5
REF_DELTA = 5

MOVE_DIST_REF = 10.0
MOVE_DIST_SMALL_REF = 5.0
MOVE_DIST_SMALL = 5.0

class Dragger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None
        self.last_ref = None
        self.moving = False
        self.points = []

    def start(self):
        self.reset()

    def check_move(self, pt, ptref):
        if self.last_ref is None:
            self.last_ref = ptref
            self.last = pt
            return
        d = dist(ptref, self.last_ref)
        if d > MOVE_DIST_REF:
            self.moving = True
            self.points.append(self.last)
            self.last = pt
            self.last_ref = ptref

    def valid_move(self, pt, ptref):
        dx1 = ptref.x - self.last_ref.x
        dx2 = pt.x - self.last.x
        dy1 = ptref.y - self.last_ref.y
        dy2 = pt.y - self.last.y

        valid_x = (dx1 <= 0 and dx2 <= 0) or (dx1 >= 0 and dx2 >= 0)
        valid_y = (dy1 <= 0 and dy2 <= 0) or (dy1 >= 0 and dy2 >= 0)
        return valid_x and valid_y

    def handle_move(self, pt, ptref):
        d = dist(ptref, self.last_ref)
        if d > MOVE_DIST_SMALL_REF:
            d = dist(pt, self.last)
            if d > MOVE_DIST_SMALL:
                self.points.append(pt)
                self.last = pt
                self.last_ref = ptref

    def track(self, pt, ptref):
        if self.moving:
            self.handle_move(pt, ptref)
        else:
            self.check_move(pt, ptref)
