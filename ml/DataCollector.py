import math
import time
import json

DATA_PATH = 'data/nomove'

class DataCollector:
    def __init__(self):
        self.started = False
        self.cur = []

    def commit(self):
        fpath = DATA_PATH + '/d_' + str(int(time.time())) + '.json'
        with open(fpath, 'w') as fd:
            fd.write(json.dumps(self.cur))

    def toggle(self):
        self.started =  not self.started
        if not self.started and len(self.cur) > 0:
            print("Finished")
            self.commit()
        else:
            print("Collecting")

    def handle(self, landmarks):
        if not self.started:
            return
        data = []
        for ld in landmarks:
            data.append({'x': ld.x, 'y': ld.y, 'z': ld.z})
        self.cur.append(data)