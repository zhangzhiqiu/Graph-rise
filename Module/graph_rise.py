import tensorflow as tf
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet
from load.util import load_image, load_test, load_graph , load_bird, load_fossil
from Module.model_layer import embedding_layer, output_layer,output_layer1
from Module.loss import graph_loss, predict_loss
import argparse
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model
import os
import csv

def build_model(model, head_model, out_model, input_shape=[224,224,3]):
    x1 = input1 = tf.keras.Input(shape=input_shape, name='input1')
    x2 = input2 = tf.keras.Input(shape=input_shape, name='input2')

    x1 = model(x1)
    emb1 = head_model(x1)
    out = out_model(emb1)

    x2 = model(x2)
    emb2 = head_model(x2)

    emb = emb1 - emb2

    return tf.keras.Model(inputs=[input1, input2],outputs=[out, emb])



def graph_rise_compile(args, shape):
    # optizem
    sgd = tf.keras.optimizers.SGD(lr=args.lr, decay=0.00004, momentum=0.9)#, clipvalue=0.5
    print('optimizer_initialize_finish')

    #build_model
    if args.gpu_num==1:
        sgd = tf.keras.optimizers.SGD(lr=args.lr, clipvalue=0.5, decay=0.00004, momentum=0.9)  # , clipvalue=0.5
        embedding_ly = embedding_layer([7, 7, 2048], args.embed_dim)
        output_ly = output_layer1(args.embed_dim, shape)  # [7, 7, 2048]
        ResNet = resnet.ResNet101(include_top=False, weights=None, input_shape=(224, 224, 3))  # ResNet

        if args.pretrain:
            ResNet.load_weights('./save/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5')
        body_model = build_model(ResNet, embedding_ly, output_ly)

        if args.resume:
            body_model.load_weights(filepath=args.save_path)
        parallel_model = body_model
        print('build_model_finish')

        parallel_model.compile(optimizer=sgd,
                               loss={'output_layer': 'categorical_crossentropy', 'tf_op_layer_Sub': graph_loss},
                               loss_weights={'output_layer': 1, 'tf_op_layer_Sub': args.alpha},
                               metrics=['accuracy'])  # predict_loss'categorical_crossentropy'
        print('compile_finish')

    else:
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:3"])#, "/gpu:1", "/gpu:2", "/gpu:3
        with strategy.scope():
            embedding_ly = embedding_layer([7, 7, 2048], args.embed_dim)
            output_ly = output_layer1(args.embed_dim, shape)  # [7, 7, 2048]
            ResNet = resnet.ResNet101(include_top=False, weights=None, input_shape=(224, 224, 3))  # ResNet
            #load_pretrain_model
            if args.pretrain:
                ResNet.load_weights('./save/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5')
            body_model = build_model(ResNet, embedding_ly, output_ly)
            # load_check_point
            if args.resume:
                body_model.load_weights(filepath=args.save_path)
            parallel_model = multi_gpu_model(body_model, gpus=3)

            parallel_model.compile(optimizer=sgd ,loss={'output_layer':'categorical_crossentropy','tf_op_layer_Sub':graph_loss},
                               loss_weights={'output_layer':1,'tf_op_layer_Sub':args.alpha} ,
                               metrics=['accuracy'])#predict_loss'categorical_crossentropy'

    return parallel_model

