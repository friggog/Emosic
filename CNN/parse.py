import csv
import os

import cv2
import numpy as np


def get_subimage(image, centre, theta, w, h):
    if h > w:
        theta += 90
        t = h
        h = w
        w = t
    rot = cv2.getRotationMatrix2D(centre, theta, 1)
    rotated = cv2.warpAffine(image, rot, image.shape[:2])
    out = cv2.getRectSubPix(rotated, (w, h), centre)
    return out


def make_box(points, image, b=1):
    rect = cv2.minAreaRect(np.array(points))
    ((x, y), (w, h), a) = rect
    if w > h:
        w, h = int(w * b), int(w * 0.5 * b)
    else:
        w, h = int(h * 0.5 * b), int(h * b)
    rect = ((x, y), (w, h), a)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return get_subimage(image, (x, y), a, w, h)


def mouth_int(path, landmarks):
    path = 'data/' + path
    image = cv2.imread(path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(make_box(landmarks[48:68], grey, b=2))


def load_data(p, limit=-1):
    print('LOADING', p)
    labelset = []
    featureset = []
    with open(p, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in r:
            if 'sub' in row[0]:
                continue
            try:
                valence = float(row[-2])
                arousal = float(row[-1])
                emotion = int(row[-3])
                if (arousal == -2 or valence == -2) and emotion > 7:
                    continue
                x, y, w, h = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]  # x, y, w, h
                path = 'data/' + row[0]
                o_path = 'data_p/' + row[0]
                if not os.path.exists(o_path):
                    image = cv2.imread(path)[int(y):int(y + h), int(x):int(x + w)]
                    image = cv2.resize(image, (256, 256))
                    o_dir = '/'.join(o_path.split('/')[:-1])
                    if not os.path.exists(o_dir):
                        os.makedirs(o_dir)
                    cv2.imwrite(o_path, image)
                count += 1
                print('Processed:', count, end="\r")
                if limit > 0 and count == limit:  # TEMP
                    break
            except Exception:
                continue
    print('\n')
    return featureset, labelset


train_f, train_l = load_data('training.csv')
val_f, val_l = load_data('validation.csv')
