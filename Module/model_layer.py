import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet
from tensorflow import keras
from load.util import load_image, load_test
import os



def embedding_layer(input_shape, out_dim, name='embedding_layer'):

    x = input = keras.Input(shape=input_shape)
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dense(out_dim, activation='relu',name='ebed_layer')(x)

    return keras.Model(inputs=input, outputs=x, name=name)


def output_layer(input_shape, out_dim, name='output_layer'):

    x = input = keras.Input(shape=input_shape)
    x = keras.layers.GlobalAveragePooling2D(name='out_pool')(x)
    x = keras.layers.Dense(out_dim, activation='softmax', name='out_layer')(x)

    return keras.Model(inputs=input, outputs=x, name=name)

def output_layer1(input_shape, out_dim, name='output_layer'):

    x = input = keras.Input(shape=input_shape)
    x = keras.layers.Dense(out_dim, activation='softmax', name='out_layer')(x)

    return keras.Model(inputs=input, outputs=x, name=name)
