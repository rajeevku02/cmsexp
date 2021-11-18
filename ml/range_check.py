import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    inputs = keras.Input(shape=(1))
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[inputs], outputs=[x])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_data():
    inp = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    labs = np.array([[0], [0], [1], [1], [1], [0], [0], [0], [0], [0]])
    return inp, labs

def main():
    print("creating model")
    model = create_model()
    print("creating data")
    inp, labs = create_data()
    print("fit...")
    model.fit(inp, labs, batch_size=10, epochs=1000)
    print(model.predict(inp))
    print(model.predict(np.array([100])))
    print(model.predict(np.array([-100])))
    print(model.predict(np.array([1000])))
    print(model.predict(np.array([3.5])))
    print(model.predict(np.array([4.5])))
    print(model.predict(np.array([5.5])))
    print(model.predict(np.array([6.5])))
    
main()