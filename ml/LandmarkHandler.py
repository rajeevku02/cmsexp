import time
import math

# (720, 1280, 3)
MIN_DIST = 7.0

MOVE_DIST = 10.0
MOVE_DIST_SMALL = 5.0
from Util import dist, dist2, pt

STOP_SEC = 0.4

CACHE_SZ = 4

DIST1 = 55
DIST2 = 90

class LandmarkHandler:
    def __init__(self):
        self.lastp0 = None
        self.moving = False
        self.lastts = None
        self.pts1 = []
        self.pts2 = []
        self.clicked = False
        self.verbose = False

    def log(self, msg):
        if self.verbose:
            print(msg)

    def toggle(self):
        self.verbose = not self.verbose

    def handle(self, landmarks):
        self.check_move(pt(landmarks[0]))
        self.check_dist(landmarks)
        # self.check(pt(landmarks[4]), pt(landmarks[8]))

    def check_dist(self, landmarks):
        #self.check_click(pt(landmarks[4]), pt(landmarks[8]))
        self.check_click(pt(landmarks[4]), pt(landmarks[10]))
        
        #d1 = dist2(pt(landmarks[4]), pt(landmarks[8]))
        #d2 = dist(pt(landmarks[4]), pt(landmarks[8]))
        #print("d1=", d1, " - ", d2)
        #d1 = dist2(pt(landmarks[4]), pt(landmarks[10]))
        #d2 = dist(pt(landmarks[4]), pt(landmarks[10]))
        #print("d2=", d1, " - ", d2)

    def check_click(self, p1, p2):
        d = dist(p1, p2)
        self.log(d)
        if self.clicked:
            if d > DIST2:
                self.clicked = False
                print("UP ", d)
        elif d < DIST1:
            self.clicked = True
            print("DOWN ", d)

    def check_move(self, p):
        #breakpoint()
        if self.lastp0 == None:
            self.lastp0 = p
            self.lastts = time.time()
            return
        ts = time.time()
        if self.moving:
            d = dist(p, self.lastp0)
            #print(d)
            if d < MOVE_DIST_SMALL:
                if ts - self.lastts > STOP_SEC:
                    #print("Stopped moving")
                    self.moving = False
                    self.lastp0 = None
                    self.lastts = None
            else:
                self.move(p)
        else:
            d = dist(p, self.lastp0)
            if d > MOVE_DIST:
                self.moving = True
                #print("Started moving")
                self.move(self.lastp0)
                self.move(p)

    def move(self, p):
        self.lastp0 = p
        self.lastts = time.time()

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
        