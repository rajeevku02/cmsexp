import numpy as np
from tensorflow import keras
from Gestures import *
from Util import dist, log, pt

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
        
        self.drag1_active = False
        self.drag2_active = False
        
    def predict(self, landmarks):
        arr = []
        for item in landmarks:
            arr.append(item.x)
            arr.append(item.y)
            arr.append(item.z)
        #breakpoint()
        out = self.model.predict(np.array(arr).reshape([1, -1]))
        mx = np.argmax(out, axis=-1)
        idx = int(mx[0]) 
        print(gestures_names[idx])
        return idx

    def get(self, landmarks):
        #self.info(landmarks)
        idx = self.predict(landmarks)
        pts = [pt(p) for p in landmarks]
        ges = self.check_drag2(idx, pts)
        if ges is not None:
            self.drag1_active = False
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
            self.drag2_active = False
            return None
        d1 = dist(pts[4], pts[10])
        d2 = dist(pts[10], pts[11])
        THRESHOLD_1 = 1.5
        THRESHOLD_2 = 1.4
        factor = 100.0
        if d1 != 0:
            factor = d2 / d1
        if self.drag2_active:
            if factor < THRESHOLD_2:
                self.drag2_active = False
                return None
            else:
                return self.drag2_gesture
        else:
            if factor > THRESHOLD_1:
                self.drag2_active = True
                return self.drag2_gesture
            else:
                return None

    def info(self, landmarks):
        pts = [pt(p) for p in landmarks]
        d1 = dist(pts[4], pts[10])
        d2 = dist(pts[10], pts[11])
        if d1 != 0:
            val = d2 / d1
            if val > 1.5:
                print(val, " YES")
            else:
                 print(val, "NO")
            #print (val)
        #print("z ", pts[9].z, ' , ', pts[10].z)