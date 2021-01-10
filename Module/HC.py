import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet
from load.util import load_image, load_test, load_graph, load_bird, load_fossil
from Module.model_layer import embedding_layer, output_layer, output_layer1
from Module.loss import graph_loss, predict_loss
import argparse
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model
import os
import csv



def HC_GE_compile(args,input_shape,out_shape1,out_shape2):
    # optizem
    sgd = tf.keras.optimizers.SGD(lr=args.lr, clipvalue=0.5, decay=0.00004, momentum=0.9)  # , clipvalue=0.5
    print('optimizer_initialize_finish')

    # build_model
    X = input1 = tf.keras.Input(shape=input_shape, name='input1')


    with tf.device('/gpu:0'):
        ResNet_GE = resnet.ResNet101(include_top=False, weights='imagenet', input_shape=(args.image_size, args.image_size, 3))  # ResNet

        GE_pre = ResNet_GE(X)
        GE_pre = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_GE')(GE_pre)
        GE_pre = tf.keras.layers.Dense(out_shape1, activation='relu', name='out_GE_midel')(GE_pre)
        out1 = tf.keras.layers.Dense(out_shape1, activation='softmax',name='out_GE')(GE_pre)

        HC_net = tf.keras.Model(inputs=[input1], outputs=[out1])
        if args.resume:
            HC_net.load_weights(filepath=args.save_path)
        print('build_model_finish')




    HC_net.compile(optimizer=sgd,
                           loss={'out_GE': 'categorical_crossentropy'},
                           loss_weights={'out_GE': 1},
                           metrics=['accuracy'])  # predict_loss'categorical_crossentropy'
    print('compile_finish')
    return HC_net

def HC_SP_compile(args,input_shape,out_shape1,out_shape2):
    # optizem
    sgd = tf.keras.optimizers.SGD(lr=args.lr,  decay=0.00004, momentum=0.9)  # , clipvalue=0.5
    print('optimizer_initialize_finish')

    # build_model
    X = input1 = tf.keras.Input(shape=input_shape, name='input1')

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    with strategy.scope():
        ResNet_SP = resnet.ResNet101(include_top=False, weights='imagenet', input_shape=(args.image_size, args.image_size, 3))

        SP_pre = ResNet_SP(X)
        SP_pre = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_SP')(SP_pre)
        #SP_pre = tf.keras.layers.Dense(out_shape2, activation='relu', name='out_SP_midel')(SP_pre)
        out2 = tf.keras.layers.Dense(out_shape2, activation='softmax', name='out_SP')(SP_pre)

        HC_net = tf.keras.Model(inputs=[input1], outputs=[out2])

        if args.resume:
            HC_net.load_weights(filepath=args.save_path)
        print('build_model_finish')

        HC_net = multi_gpu_model(HC_net, gpus=2)

        HC_net.compile(optimizer=sgd,
                               loss={'out_SP': 'categorical_crossentropy'},
                               loss_weights={'out_SP': 1},
                               metrics=['accuracy'])  # predict_loss'categorical_crossentropy'
        print('compile_finish')
    return HC_net


def HC_T_compile(args,input_shape,out_shape1,out_shape2):
    # optizem
    sgd = tf.keras.optimizers.SGD(lr=args.lr, decay=0.00004, momentum=0.9)  # , clipvalue=0.5
    print('optimizer_initialize_finish')

    # build_model
    X = input1 = tf.keras.Input(shape=input_shape, name='input1')
    Y1 = input2 = tf.keras.Input(shape=out_shape2, name='input2')


    with tf.device('/gpu:0'):
        ResNet_GE = resnet.ResNet101(include_top=False, weights='imagenet', input_shape=(args.image_size, args.image_size, 3))  # ResNet

        GE_pre = ResNet_GE(X)
        GE_pre = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_GE')(GE_pre)
        out1 = tf.keras.layers.Dense(out_shape1, activation='softmax',name='out_GE')(GE_pre)

    with tf.device('/gpu:1'):
        ResNet_SP = resnet.ResNet152(include_top=False, weights='imagenet', input_shape=(args.image_size, args.image_size, 3))

        SP_pre = ResNet_SP(X)
        SP_pre = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_SP')(SP_pre)

        SP_pre = tf.keras.layers.Dense(out_shape2, activation='relu', name='out_SP_midel')(SP_pre)#_midel
        GE_pre_relu = tf.keras.layers.Dense(out_shape2, activation='relu')(GE_pre)

        SP_out = tf.keras.layers.multiply([GE_pre_relu, SP_pre])
        out2 = tf.keras.layers.Dense(out_shape2, activation='softmax', name='out_SP')(SP_out)

        HC_net = tf.keras.Model(inputs=[input1], outputs=[out1, out2])



        if args.resume:
            HC_net.load_weights(filepath=args.save_path)
        print('build_model_finish')




    HC_net.compile(optimizer=sgd,
                           loss={'out_GE': 'categorical_crossentropy', 'out_SP': 'categorical_crossentropy'},
                           loss_weights={'out_GE': 1, 'out_SP': 1},
                           metrics=['accuracy'])  # predict_loss'categorical_crossentropy'
    print('compile_finish')
    return HC_net