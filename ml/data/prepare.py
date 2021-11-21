import os
import json
import random

'''
0: drag1
1: drag2
2: thumb
3: pinch
4: thumb_index
5: open
6: other
'''

data = []

def load_file(folder, label):
    files = os.listdir(folder)
    for f in files:
        fname = folder + '/' + f
        print('loading ', fname)
        with open(fname) as fd:
            ct = fd.read()
        doc = json.loads(ct)
        for item in doc:
            data.append({'x': item, 'y': label})

def load_data():
    load_file('./drag1', [1, 0, 0, 0, 0, 0, 0])
    load_file('./drag2', [0, 1, 0, 0, 0, 0, 0])
    load_file('./thumb', [0, 0, 1, 0, 0, 0, 0])
    load_file('./pinch', [0, 0, 0, 1, 0, 0, 0])
    load_file('./thumb_index', [0, 0, 0, 0, 1, 0, 0])
    load_file('./open', [0, 0, 0, 0, 0, 1, 0])
    load_file('./other', [0, 0, 0, 0, 0, 0, 1])

def main():
    load_data()
    random.shuffle(data)
    random.shuffle(data)
    print("Len = ", len(data))
    with open('gestures.json', 'w') as fd:
        fd.write(json.dumps(data, indent=2))
main()