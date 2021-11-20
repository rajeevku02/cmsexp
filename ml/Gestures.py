from Dragger import Dragger
from DrawShapes import add_points

class Gesture:
    def __init__(self, id):
        self.dragger = Dragger()
        self.id = id
    
    def done(self):
        pass

    def move(self, landmarks):
        pass

class Drag1Gesture(Gesture):
    def __init__(self):
        super().__init__('drag1')
    
    def move(self, landmarks):
        self.dragger.track(landmarks[4], landmarks[0])

    def done(self):
        add_points(self.dragger.points)
        self.dragger.points = []

class Drag2Gesture(Gesture):
    def __init__(self):
        super().__init__('drag2')

    def move(self, landmarks):
        self.dragger.track(landmarks[8], landmarks[0])

    def done(self):
        add_points(self.dragger.points)
        self.dragger.points = []

class ThumGesture(Gesture):
    def __init__(self):
        super().__init__('thumb')

    def move(self, landmarks):
        self.dragger.track(landmarks[4], landmarks[0])

class PinchGesture(Gesture):
    def __init__(self):
        super().__init__('pinch')

    def move(self, landmarks):
        self.dragger.track(landmarks[0], landmarks[0])

