import time
from DrawShapes import undo

last_gesture = None

CLICK_GAP = 0.3
UNDO_GAP = 2.5

last_click_at = -1
last_undo_at = -1

def got_click():
    global last_click_at
    if last_click_at > 0:
        print("Double click")
        last_click_at = -1
    else:
        last_click_at = time.time()

def finish_click():
    global last_click_at
    if last_click_at > 0 and time.time() - last_click_at > CLICK_GAP:
        print("Click")
        last_click_at = -1

def got_undo():
    global last_undo_at
    if last_undo_at < 0:
        last_undo_at = time.time()
        return
    if time.time() - last_undo_at < UNDO_GAP:
        undo()
    last_undo_at = -1

def handle_gesture_transient(gesture):
    global last_gesture
    if last_gesture is None:
        last_gesture = gesture
        return
    if (last_gesture.id == 'drag1' or last_gesture.id == 'drag2') and \
        gesture.id == 'noop' and len(last_gesture.dragger.points) == 0:
        got_click()
    if last_gesture.id == 'thumb' and gesture.id == 'noop':
        got_undo()
    finish_click()
    last_gesture = gesture
