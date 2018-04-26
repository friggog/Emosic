#! /usr/local/bin/python3

import os

import numpy as np
from keras.applications.mobilenet import DepthwiseConv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GaussianDropout, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CLASSIFY = 0
REGRESS = 1

# OTPIONS #
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


def depth_conv_block(model, d, k, s):
    model.add(DepthwiseConv2D((k, k), strides=(s, s), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(d, (1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model


def mobilenet_style_model(C_or_R, dropout=0):
    model = Sequential()
    alpha = 1
    # CONV
    model.add(Conv2D(int(32 * alpha), (3, 3), padding='same', use_bias=False, strides=(2, 2), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # DEPTHWISE CONVS
    model = depth_conv_block(model, int(64 * alpha), 3, 1)

    model = depth_conv_block(model, int(128 * alpha), 3, 2)
    model = depth_conv_block(model, int(128 * alpha), 3, 1)

    model = depth_conv_block(model, int(256 * alpha), 3, 2)
    model = depth_conv_block(model, int(256 * alpha), 3, 1)

    model = depth_conv_block(model, int(512 * alpha), 3, 2)
    model = depth_conv_block(model, int(512 * alpha), 3, 1)
    model = depth_conv_block(model, int(512 * alpha), 3, 1)
    model = depth_conv_block(model, int(512 * alpha), 3, 1)
    model = depth_conv_block(model, int(512 * alpha), 3, 1)
    model = depth_conv_block(model, int(512 * alpha), 3, 1)

    model = depth_conv_block(model, int(1024 * alpha), 3, 2)
    model = depth_conv_block(model, int(1024 * alpha), 3, 1)
    # FLATTEN
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def vgg_style_model(C_or_R, dropout=(0.2, 0.5)):
    model = Sequential()
    # CONV BLOCK 1
    model.add(Conv2D(16, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 2
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 3
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 4
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 5
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # flatten
    model.add(Flatten())
    # dense 1
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout[1]))
    # dense 2
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout[1]))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def alexnet_style_model(C_or_R, dropout=(0.2, 0.5)):
    model = Sequential()
    # CONV BLOCK 1
    model.add(Conv2D(16, (9, 9), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 2
    model.add(Conv2D(32, (7, 7), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 3
    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 4
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # CONV BLOCK 5
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(GaussianDropout(dropout[0]))
    # FLATTEN
    model.add(Flatten())
    # DENSE 1
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout[1]))
    # DENSE 2
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout[1]))
    # OUTPUT
    if C_or_R == CLASSIFY:
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def regressor_from_classifier(model, drop=False):
    # REMOVE CLASSIFIER LAYER
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    # ADD REGRESSOR OUTPUT
    if drop:
        model.add(Dropout(0.3))
    model.add(Dense(2, activation='linear', name='regressor_output'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def load_and_save(model, m):
    model.load_weights(m + '.h5')
    for layer in model.layers:
        if type(layer) is Dropout or type(layer) is GaussianDropout:
            model.layers.remove(layer)
    model.save(m + '_out.h5')


def visualise(model, name):
    plot_model(model, to_file=name + '.png', show_shapes=True, show_layer_names=False)


def train(C_or_R, model, output_path, epochs, batch_size):
    print('** LOADING DATA **')
    t_paths = np.load('training_paths.npy')
    t_labels = np.load('training_labels.npy')
    t_paths, t_labels, t_weights = process_data(C_or_R, t_paths, t_labels)
    v_paths = np.load('validation_paths.npy')
    v_labels = np.load('validation_labels.npy')
    v_paths, v_labels, v_weights = process_data(C_or_R, v_paths, v_labels)
    print('** FITTING MODEL **')
    t_steps = len(t_labels) // batch_size
    v_steps = len(v_labels) // batch_size
    if C_or_R == CLASSIFY:
        history = model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, batch_size),
            steps_per_epoch=t_steps,
            class_weight=t_weights,
            epochs=epochs,
            validation_data=load_images(C_or_R, v_paths, v_labels, batch_size, eval=True),
            validation_steps=v_steps,
            callbacks=[ModelCheckpoint(output_path + '_T.h5', monitor='val_acc', save_best_only=True)])
    else:
        history = model.fit_generator(
            load_images(C_or_R, t_paths, t_labels, batch_size),
            steps_per_epoch=t_steps,
            epochs=epochs,
            validation_data=load_images(C_or_R, v_paths, v_labels, batch_size, eval=True),
            validation_steps=v_steps,
            callbacks=[ModelCheckpoint(output_path + '_T.h5', save_best_only=True)])
    print('** EXPORTING MODEL **')
    np.save(output_path + '_HIST', history.history)
    for layer in model.layers:
        if type(layer) is Dropout or type(layer) is GaussianDropout:
            model.layers.remove(layer)
    model_json = model.to_json()
    with open(output_path + '_ARCH.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(output_path + '_WEIGHTS.h5')
    model.save(output_path + '_FULL.h5')


if __name__ == '__main__':
    print('Haven\'t got around to a CLI for this')
    print('Choose a model from mobilenet_style_model, vgg_style_model, alexnet_style_model')
    print('C_or_R means classifier or regressor')
    print('Call train() on the model with appropriate C_or_R, output and train options')
    print('A pretrained classifier can be loaded and passed to regressor_from_classifier() to get a regression model which can be trained as normal with C_or_R=REGRESS')
    print('load_and_save() is used to trim unneeded data from a model to get the true output model, give it a fresh initialization of the right mdoel type and path to the trained model')
