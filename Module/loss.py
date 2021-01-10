import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet
from tensorflow import keras
from load.util import load_image, load_test
import os

def predict_loss(y_true,y_pred):
    loss = tf.losses.mean_squared_error(y_pred,y_true)#tf.reduce_mean(tf.square(y_pred - y_true))
    return loss


def graph_loss(graph, embedding):
    return tf.reduce_mean(graph*tf.reduce_mean(tf.square(embedding),axis=1))







    '''
    print('11111111111111111111111111')
    print(graph.get_shape())
    print(embedding.get_shape())
    mask1 = tf.ones([64, 64])
    mask2 = tf.eye(64)
    mask = mask1 - mask2

    A = graph
    At= tf.transpose(A, perm=[1, 0])
    Neiber = tf.matmul(A,At)
    Neiber = Neiber*mask

    B  = embedding
    Bt = tf.transpose(A, perm=[1, 0])
    B_ = tf.matmul(B,Bt)
    B1 = tf.matmul(B_*mask2, mask1)
    B2 = tf.matmul(mask1, B_*mask2)

    ebed_l2 = B1 + B2 - 2*B_

    graph_l = tf.reduce_mean(Neiber*ebed_l2)
    '''
