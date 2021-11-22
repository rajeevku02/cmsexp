import time
from GestureRecognizer import GestureRecognizer
from Util import pt
from DrawShapes import draw_shapes

PENDING_THRESHOLD = 0.2

class LandmarkHandler:
    def __init__(self):
        self.gesture_recognizer = GestureRecognizer()
        self.gesture = None
        self.pending_gesture = None
        self.pending_ts =time.time()

    def points(self):
        if self.gesture:
            return self.gesture.dragger.points
        return []

    def draw(self, image):
        draw_shapes(image, self.points())
        if self.pending_gesture:
            draw_shapes(image, self.pending_gesture.dragger.points)

    def handle_last(self, gesture):
        if self.pending_gesture:
            if self.pending_gesture == gesture:
                #print('Reusing gesture')
                self.pending_gesture = None
            else:
                tm = time.time()
                if tm - self.pending_ts > PENDING_THRESHOLD:
                    self.pending_gesture.done()
                    self.pending_gesture = None
        if self.gesture is not None:
            if self.gesture.id == 'drag2' and self.gesture.dragger.moving and gesture.id == 'noop':
                self.pending_gesture = self.gesture
            elif gesture.id != self.gesture.id:
                self.gesture.done()

    def handle(self, landmarks, hand):
        if hand.lower() == "right":
            return
        pts = [pt(p) for p in landmarks]
        gesture = self.gesture_recognizer.get(landmarks)
        self.handle_last(gesture)
        self.gesture = gesture
        self.gesture.move(pts)
