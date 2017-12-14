from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from math import floor
import csv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CLASSIFY = 0
REGRESS = 1


# OTPIONS #
CLASSIFY_OR_REGRESS = CLASSIFY
BATCH_SIZE = 16
EPOCHS = 8

# LOADERS #
def load_images(paths, labels, batch_size=32):
    batch_n = 0
    while True:
        batch_d = []
        batch_l = []
        for i in range(batch_size):
            if batch_n*batch_size + i > len(paths) - 1:
                batch_n = 0
            path = paths[batch_n*batch_size + i]
            img = image.load_img(path, target_size=(256, 256))
            x = image.img_to_array(img)/255
            x = image.random_rotation(x, 10)
            x = image.random_shift(x, 0.1, 0.1)
            # x = image.random_shear(x, 0.1)
            x = image.random_zoom(x, (0.1,0.1))
            if np.random.random() < 0.5:
                x = image.flip_axis(x, 1)
            y = labels[batch_n*batch_size + i]
            batch_d.append(x)
            batch_l.append(y)
        batch_d = np.array(batch_d).reshape((batch_size, 256, 256, 3))
        if CLASSIFY_OR_REGRESS == CLASSIFY:
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
            path =  'data_p/' + row[0]
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

def process_data(paths, labels):
    labels_out = []
    paths_out = []
    count = 0
    for i, (emotion, valence, arousal) in enumerate(labels):
        if CLASSIFY_OR_REGRESS == CLASSIFY:
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
    if CLASSIFY_OR_REGRESS == CLASSIFY:
        labels_out = to_categorical(labels_out, num_classes=8)
    print('Processed:', count)
    return paths_out, labels_out

# MODEL #
model = Sequential()
# TODO sort out model
model.add(Conv2D(32, (16, 16), input_shape=(256, 256, 3), activation='relu'))
model.add(Conv2D(32, (16, 16), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(64, (8, 8), activation='relu'))
model.add(Conv2D(64, (8, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
if CLASSIFY_OR_REGRESS == CLASSIFY:
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
else:
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=opt)

print('** LOADING DATA **')
t_paths = np.load('training_paths.npy')
t_labels = np.load('training_labels.npy')
t_paths, t_labels = process_data(t_paths, t_labels)
v_paths = np.load('validation_paths.npy')
v_labels = np.load('validation_labels.npy')
v_paths, v_labels = process_data(v_paths, v_labels)

print('** FITTING MODEL **')
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

model.fit_generator(
        load_images(t_paths, t_labels, BATCH_SIZE),
        steps_per_epoch=len(t_labels)/BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=load_images(v_paths, v_labels, BATCH_SIZE),
        validation_steps=len(v_labels),
        callbacks=[LearningRateScheduler(lr_schedule)])

print('** EXPORTING MODEL **')
for k in model.layers:
    if type(k) is keras.layers.Dropout:
        model.layers.remove(k)

model.save_weights('AFF_NET_'+CLASSIFY_OR_REGRESS+'.h5')

# idea is to train on emotion classification + fine tune for valence/arousal??
# for finetuning
# model.layers.pop()
# model.outputs = [model.layers[-1].output]
# model.layers[-1].outbound_nodes = []
# model.add(Dense(2, activation='linear'))
# for layer in model.layers[:TODO]:
#     layer.trainable = False
# compile + fit
