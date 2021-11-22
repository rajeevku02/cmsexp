from Dragger import Dragger
from DrawShapes import add_points, del_shapes, move_shapes

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
        self.points = []

    def move(self, landmarks):
        self.points.append(landmarks[4])
        if len(self.points) > 1:
            del_shapes(self.points)
            self.points = []

    def done(self):
        self.points = []

class PinchGesture(Gesture):
    def __init__(self):
        super().__init__('pinch')
        self.points = []

    def move(self, landmarks):
        self.points.append(landmarks[0])
        if len(self.points) > 1:
            move_shapes(self.points)
            del self.points[:-1]

    def done(self):
        self.points = []

