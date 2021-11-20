from Gestures import *
from Util import dist, log

DIST1 = 55
DIST2 = 90

class GestureRecognizer:
    def __init__(self):
        self.clicked = False
        self.drag_gesture = Drag1Gesture()
        self.noop_gesture = Gesture('noop')

    def get(self, landmarks):
        self.check_click(landmarks[4], landmarks[8])
        if self.clicked:
            return self.drag_gesture 
        return self.noop_gesture
    
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