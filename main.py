import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet
from load.util import load_fossil,load_HC_fossil
from Module.model_layer import embedding_layer, output_layer,output_layer1
from Module.loss import graph_loss, predict_loss
from Module.HC import HC_T_compile,HC_GE_compile,HC_SP_compile
import argparse
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model
import os
import csv


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def get_next_batch(pointer,train_images):
    image_batch = []
    images = train_images[pointer * args.batch_size:(pointer + 1) * args.batch_size]

    for img in range(args.batch_size):
        arr = images[img, :, :, :]
        arr = np.array(arr)
        arr = arr.astype('float32') / 127.5 - 1

        image_batch.append(arr)
    return image_batch


def next_batch_labels(pointer,shuffled_labels):
    labels_batch = []
    labels = shuffled_labels[pointer * args.batch_size:(pointer + 1) * args.batch_size]
    return labels

#

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])

    # returned the smoothed labels
    return labels

def scheduler(epoch):
    if epoch<10:
        lr = 0.001
    else:
        lr = 0.001 * 0.1**(float(epoch)//10)
    return lr



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='./data/test1',type=str)
    parser.add_argument('--save_path', default='./save/checkpoint.h5',type=str)
    parser.add_argument('--graph_path', default='./data/attrann.mat', type=str)

    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--pretrain', default=False, action='store_true')

    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--embed_dim', default=4096, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()

    #load_data
    if not args.train:
        train_images,train_labels_SP,train_labels_GE,sp_ge = load_HC_fossil(args.file_path, image_size=args.image_size,train=0)
        '''test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow_from_directory(
            os.path.join(args.file_path, 'test_images'),
            target_size=(args.image_size, args.image_size),
            batch_size=32)'''


    else:
        train_images, train_labels_SP, train_labels_GE, sp_ge = load_HC_fossil(args.file_path,
                                                                               image_size=args.image_size, train=1)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        train_datagen.fit(train_images)

        test_images, test_labels_SP, test_labels_GE, _ = load_HC_fossil(args.file_path,
                                                                               image_size=args.image_size, train=0)

        #train_images,train_labels_SP,train_labels_GE,sp_ge = load_HC_fossil(args.file_path, image_size=args.image_size,train=1)
        #train_labels_SP = smooth_labels(train_labels_SP, 0.1)
        #train_labels_GE = smooth_labels(train_labels_GE, 0.1)


    #callback
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_path, monitor='val_loss', verbose=1,
                                                    save_best_only=False,
                                                    save_weights_only=True, mode='auto', period=1)
    print('callback_build_finish')

    #model_compile
    parallel_model = HC_SP_compile(args, [args.image_size, args.image_size, 3], 0,113)

    #train
    if args.train:
        '''parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_datagen),
            epochs=300,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint, reduce_lr],
            initial_epoch=0)'''
        parallel_model.fit_generator(train_datagen.flow(train_images, train_labels_SP, batch_size=32),
                                 steps_per_epoch=len(train_images) / 32,
                                 epochs=300,
                                 validation_data=train_datagen.flow(test_images, test_labels_SP, batch_size=32),
                                 validation_steps=100,
                                 callbacks=[checkpoint, reduce_lr],
                                 initial_epoch=0)
    #test
    else:
        #parallel_model.evaluate({'input1': train_images},{'out_GE': train_labels_GE, 'out_SP': train_labels_SP}, verbose=1)
        #parallel_model.evaluate({'input1': train_images},{'out_GE': train_labels_GE},verbose=1)

        parallel_model.evaluate({'input1': train_images},{'out_SP': train_labels_SP},verbose=1)







