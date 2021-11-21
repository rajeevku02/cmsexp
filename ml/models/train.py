import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models import get_model_1

TEST_SZ = 1000

def make_data(items):
    x = []
    y = []
    for item in items:
        arr = []
        for v in item['x']:
            arr.append(v['x'])
            arr.append(v['y'])
        x.append(arr)
        y.append(item['y'])
    return (np.array(x), np.array(y))

def get_data():
    print('Loading data')
    with open('../data/gestures.json') as fd:
        doc = json.loads(fd.read())
    print('splitting data')
    x1, y1 = make_data(doc[0: -TEST_SZ])
    x2, y2 = make_data(doc[TEST_SZ + 1:])
    print('data loaded')
    return (x1, y1, x2, y2)

def main():
    model = get_model_1()
    train_x, train_y, test_x, test_y = get_data()
    print('training')
    model.fit(train_x, train_y, batch_size=train_x.shape[0], epochs=5000)

    print('')    
    loss, acc = model.evaluate(test_x, test_y)
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)

    model.save("trained_model")

main()