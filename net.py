import csv
import os
from math import floor

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling3D)
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CLASSIFY = 0
REGRESS = 1

# OTPIONS #
BATCH_SIZE = 256
EPOCHS = 24

def load_images(C_or_R, paths, labels, batch_size=32):
    batch_n = 0
    while True:
        batch_d = []
        batch_l = []
        for i in range(batch_size):
            if batch_n * batch_size + i > len(paths) - 1:
                batch_n = 0
            path = paths[batch_n * batch_size + i]
            img = image.load_img(path, target_size=(96, 96))
            x = image.img_to_array(img) / 255
            x = image.random_rotation(x, 20)
            x = image.random_shift(x, 0.1, 0.1)
            if np.random.random() < 0.5:
                x = image.flip_axis(x, 1)
            y = labels[batch_n * batch_size + i]
            batch_d.append(x)
            batch_l.append(y)
        batch_d = np.array(batch_d).reshape((batch_size, 96, 96, 3))
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
                # ignore invalid values
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


def base_model(C_or_R, denseSize):
    model = Sequential()
    # CONV BLOCK 1
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV BLOCK 2
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV BLOCK 3
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FLATTEN
    model.add(Flatten())
    # DENSE 1
    model.add(Dense(denseSize))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # DENSE 2
    model.add(Dense(denseSize))
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


def fine_tune_model(C_or_R):
    if C_or_R == REGRESS:
        model = base_model(C_or_R, 256)
        model.load_weights('AFF_NET_C_WIP.h5')
        # REMOVE CLASSIFIER LAYER
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        # OUTPUT
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        model = base_model(C_or_R, 512)
        model.load_weights('OUT_512D.h5')
        # REMOVE DENSE LAYERS
        for i in range(9):
            model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        # DENSE 1
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # # DENSE 2
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # OUTPUT
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(C_or_R, mode=0):
    if mode == 0:
        model = base_model(C_or_R)
    else:
        model = fine_tune_model(C_or_R)

    print('** LOADING DATA **')
    t_paths = np.load('training_paths.npy')
    t_labels = np.load('training_labels.npy')
    t_paths, t_labels, t_weights = process_data(C_or_R, t_paths, t_labels)
    v_paths = np.load('validation_paths.npy')
    v_labels = np.load('validation_labels.npy')
    v_paths, v_labels, v_weights = process_data(C_or_R, v_paths, v_labels)

    print('** FITTING MODEL **')
    if C_or_R == CLASSIFY:
        ns = 'C'
        model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, BATCH_SIZE),
            steps_per_epoch=len(t_labels) // BATCH_SIZE,
            class_weight=t_weights,
            epochs=EPOCHS,
            validation_data=load_images(C_or_R, v_paths, v_labels, BATCH_SIZE),
            validation_steps=len(v_labels) // BATCH_SIZE,
            callbacks=[ModelCheckpoint('AFF_NET_C_WIP.h5', save_best_only=True)])
    else:
        ns = 'R'
        model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, BATCH_SIZE),
            steps_per_epoch=len(t_labels) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=load_images(C_or_R, v_paths, v_labels, BATCH_SIZE),
            validation_steps=len(v_labels) // BATCH_SIZE,
            callbacks=[ModelCheckpoint('AFF_NET_R_WIP.h5', save_best_only=True)])

    print('** EXPORTING MODEL **')
    model_json = model.to_json()
    with open('AFF_NET_' + ns + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('AFF_NET_' + ns + '.h5')

train(CLASSIFY, mode=1)
train(REGRESS, mode=1)