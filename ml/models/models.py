import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model_1():
    inputs = keras.Input(shape=(63))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(7, activation="softmax")(x)
    model = keras.Model(inputs=[inputs], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_model_2():
    inputs = keras.Input(shape=(42))
    x = layers.Dense(128, activation="relu")(inputs)
    #x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(7, activation="softmax")(x)
    model = keras.Model(inputs=[inputs], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 0.9989 0.9977 | 264 264 7