from Gestures import *
from Util import dist, log

DIST1 = 55
DIST2 = 90

class GestureRecognizer:
    def __init__(self):
        self.clicked = False

    def get(self, landmarks):
        self.check_click(landmarks[4], landmarks[8])
        if self.clicked:
            return Drag1Gesture()
        return Gesture('noop')
    
    def check_click(self, p1, p2):
        d = dist(p1, p2)
        log(d)
        if self.clicked:
            if d > DIST2:
                self.clicked = False
                print("UP ", d)
        elif d < DIST1:
            self.clicked = True
            print("DOWN ", d)