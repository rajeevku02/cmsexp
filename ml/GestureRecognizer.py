import numpy as np
from tensorflow import keras
from Gestures import *

from geometry import dist
from Util import log, pt
from drag_2_gesture import check_drag_2, deactivate_drag2
from drag_1_gesture import deactivate_drag1

gestures_names = {
    0: 'drag1',
    1: 'drag2',
    2: 'thumb',
    3: 'pinch',
    4: 'thumb_index',
    5: 'open',
    6: 'other'
}

class GestureRecognizer:
    def __init__(self):
        self.model = keras.models.load_model('models/trained_model')

        self.drag_gesture = Drag1Gesture()
        self.drag2_gesture = Drag2Gesture()
        self.thumb_gesture = ThumGesture()
        self.pinch_gesture = PinchGesture()
        self.noop_gesture = Gesture('noop')
        
    def predict(self, landmarks):
        arr = []
        for item in landmarks:
            arr.append(item.x)
            arr.append(item.y)
            arr.append(item.z)
        out = self.model.predict(np.array(arr).reshape([1, -1]))
        mx = np.argmax(out, axis=-1)
        idx = int(mx[0]) 
        print(gestures_names[idx])
        return idx

    def get(self, landmarks):
        idx = self.predict(landmarks)
        pts = [pt(p) for p in landmarks]
        ges = self.check_drag2(idx, pts)
        if ges is not None:
            deactivate_drag1()
            return ges
        ges = self.check_drag1(idx, pts)
        if ges is not None:
            return ges
        ges = self.check_thumb(idx, pts)
        if ges is not None:
            return ges
        ges = self.check_pinch(idx, pts)
        if ges is not None:
            return ges
        return self.noop_gesture

    def check_pinch(self, idx, pts):
        if idx == 3:
            return self.pinch_gesture
        return None

    def check_thumb(self, idx, pts):
        if idx == 2:
            return self.thumb_gesture
        return None

    def check_drag1(self, idx, pts):
        return None

    def check_drag2(self, idx, pts):
        if not (idx == 1 or idx == 2 or idx == 3):
            deactivate_drag2()
            return None

        if check_drag_2(pts):
            return self.drag2_gesture
        return None
