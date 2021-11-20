from GestureRecognizer import GestureRecognizer
from Util import pt
from DrawShapes import draw_shapes

class LandmarkHandler:
    def __init__(self):
        self.gesture_recognizer = GestureRecognizer()
        self.gesture = None

    def points(self):
        if self.gesture:
            return self.gesture.dragger.points
        return []

    def draw(self, image):
        draw_shapes(image, self.points())

    def handle(self, landmarks, hand):
        if hand.lower() == "right":
            return
        pts = [pt(p) for p in landmarks]
        gesture = self.gesture_recognizer.get(pts)
        if self.gesture is not None:
            if gesture.id != self.gesture.id:
                self.gesture.done()
        self.gesture = gesture
        self.gesture.move(pts)
