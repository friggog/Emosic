#! /usr/local/bin/python3

import csv
import os
from math import floor

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, Flatten, GaussianDropout, GlobalAveragePooling2D, MaxPooling2D, MaxPooling3D, SeparableConv2D
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from sklearn.utils import class_weight
from keras.applications.mobilenet import DepthwiseConv2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CLASSIFY = 0
REGRESS = 1

# OTPIONS #
BATCH_SIZE = 400 # VGG/ALEX: 400
EPOCHS = 24
IMAGE_SIZE = 128


def load_images(C_or_R, paths, labels, batch_size=32, eval=False):
    batch_n = 0
    while True:
        batch_d = []
        batch_l = []
        for i in range(batch_size):
            if batch_n * batch_size + i > len(paths) - 1:
                batch_n = 0
            path = paths[batch_n * batch_size + i]
            img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            x = image.img_to_array(img) / 255
            if not eval:
                x = image.random_rotation(x, 20)
                x = image.random_shift(x, 0.1, 0.1)
                if np.random.random() < 0.5:
                    x = image.flip_axis(x, 1)
            y = labels[batch_n * batch_size + i]
            batch_d.append(x)
            batch_l.append(y)
        batch_d = np.array(batch_d).reshape((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        if C_or_R == CLASSIFY:
            batch_l = np.array(batch_l).reshape((batch_size, 8))
        else:
            batch_l = np.array(batch_l).reshape((batch_size, 2))
        yield (batch_d, batch_l)
        batch_n += 1


def load_paths(p):
    labels = []
    paths = []
    count = 0
    with open(p, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in r:
            if 'sub' in row[0]:
                continue
            valence = float(row[-2])
            arousal = float(row[-1])
            emotion = int(row[-3])
            path = 'data_p/' + row[0]
            if emotion > 7 and (arousal == -2 or valence == -2):
                continue
            if not os.path.exists(path):
                print('error: no image')
                continue
            labels.append((emotion, valence, arousal))
            paths.append(path)
            count += 1
            print('Loaded:', count, end='\r')
    print('Loaded:', count)
    return paths, labels


def process_data(C_or_R, paths, labels):
    labels_out = []
    paths_out = []
    count = 0
    for i, (emotion, valence, arousal) in enumerate(labels):
        if C_or_R == CLASSIFY:
            if emotion > 7:
                # ignore invalid emotions
                continue
            labels_out.append(emotion)
            paths_out.append(paths[i])
        else:
            if arousal == -2 or valence == -2:
                # ignore invalid values
                continue
            labels_out.append([valence, arousal])
            paths_out.append(paths[i])
        count += 1
        print('Processed:', count, end='\r')
    if C_or_R == CLASSIFY:
        weights = class_weight.compute_class_weight('balanced', np.unique(labels_out), labels_out)
        weights = dict(enumerate(weights))
        labels_out = to_categorical(labels_out, num_classes=8)
    else:
        weights = None
    print('Processed:', count)
    return paths_out, labels_out, weights


def mobilenet_style_model(C_or_R):
    model = Sequential()
    # CONV BLOCK 2
    model.add(DepthwiseConv2D(32, (3, 3), padding='same', use_bias=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(DepthwiseConv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(DepthwiseConv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(DepthwiseConv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(DepthwiseConv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # FLATTEN
    model.add(GlobalAveragePooling2D())
    # DENSE 1
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # DENSE 2
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def vgg_style_model(C_or_R):
    model = Sequential()
    # CONV BLOCK 1
    model.add(Conv2D(16, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 2
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 3
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 4
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 5
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # FLATTEN
    model.add(Flatten())
    # DENSE 1
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # DENSE 2
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def alexnet_style_model(C_or_R):
    model = Sequential()
    # CONV BLOCK 1
    model.add(Conv2D(16, (9, 9), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 2
    model.add(Conv2D(32, (7, 7), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 3
    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 4
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # CONV BLOCK 5
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(0.2))
    # FLATTEN
    model.add(Flatten())
    # DENSE 1
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # DENSE 2
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def regressor_from_classifier(model):
    # REMOVE CLASSIFIER LAYER
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    # ADD REGRESSOR OUTPUT
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def eval(name, C_or_R):
    # model = base_model(C_or_R, 256)
    model = load_model(name)
    # for layer in model.layers:
    # if type(layer) is Dropout:
    # model.layers.remove(layer)
    v_paths = np.load('validation_paths.npy')
    v_labels = np.load('validation_labels.npy')
    v_paths, v_labels, v_weights = process_data(C_or_R, v_paths, v_labels)
    res = model.evaluate_generator(load_images(C_or_R, v_paths, v_labels, BATCH_SIZE, eval=True),
                                   steps=len(v_labels) // BATCH_SIZE)
    print('Accuracy:', res[1])


def load_and_save():
    model = vgg_style_model(CLASSIFY)
    model.load_weights('M_VGG/C_T.h5')
    for layer in model.layers:
        if type(layer) is Dropout or type(layer) is GaussianDropout:
            model.layers.remove(layer)
    model.save('M_VGG/C_T_S.h5')

def visualise(model, name):
    plot_model(model, to_file=name+'.png', show_shapes=True, show_layer_names=False)

def train(C_or_R, model, name):
    print('** LOADING DATA **')
    t_paths = np.load('training_paths.npy')
    t_labels = np.load('training_labels.npy')
    t_paths, t_labels, t_weights = process_data(C_or_R, t_paths, t_labels)
    v_paths = np.load('validation_paths.npy')
    v_labels = np.load('validation_labels.npy')
    v_paths, v_labels, v_weights = process_data(C_or_R, v_paths, v_labels)
    print('** FITTING MODEL **')
    if C_or_R == CLASSIFY:
        history = model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, BATCH_SIZE),
            steps_per_epoch=len(t_labels) // BATCH_SIZE,
            class_weight=t_weights,
            epochs=EPOCHS,
            validation_data=load_images(C_or_R, v_paths, v_labels, BATCH_SIZE, eval=True),
            validation_steps=len(v_labels) // BATCH_SIZE,
            callbacks=[ModelCheckpoint(name + '_T.h5', monitor='val_acc', save_best_only=True)])
    else:
        history = model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, BATCH_SIZE, eval=True),
            steps_per_epoch=len(t_labels) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=load_images(C_or_R, v_paths, v_labels, BATCH_SIZE, eval=True),
            validation_steps=len(v_labels) // BATCH_SIZE,
            callbacks=[ModelCheckpoint(name + '_T.h5', save_best_only=True)])
    print('** EXPORTING MODEL **')
    np.save(name + '_HIST', history.history)
    for layer in model.layers:
        if type(layer) is Dropout:
            model.layers.remove(layer)
    model_json = model.to_json()
    with open(name + '_ARCH.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(name + '_WEIGHTS.h5')
    model.save(name + '_FULL.h5')


if __name__ == '__main__':
    # load_and_save()
    # eval('M_VGG/C_T_S.h5', CLASSIFY)
    # train(CLASSIFY, vgg_style_model(CLASSIFY), 'M_VGG/C')
    train(CLASSIFY, alexnet_style_model(CLASSIFY), 'M_ALEX/C')
    # train(CLASSIFY, mobilenet_style_model(CLASSIFY), 'M_MOB/C')
    # visualise(vgg_style_model(CLASSIFY), 'M_VGG/C')
