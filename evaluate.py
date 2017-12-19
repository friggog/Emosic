import os
import sys
from math import sqrt

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, mean_squared_error, roc_auc_score)
from scipy.stats import pearsonr

from net import process_data

CLASSIFY = 0
REGRESS = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def RMSE(t, p):
    return round(sqrt(mean_squared_error(t, p)), 4)


def ACC(t, p):
    return round(accuracy_score(t, p), 4)


def F1(t, p):
    return round(f1_score(t, p, average='macro'), 4)


def KAPPA(t, p):
    return round(cohen_kappa_score(t, p), 4)


def CORR(t, p):
    cc = pearsonr(t, p)[0]
    return round(cc, 4)


def AUC(t, p):
    return round(roc_auc_score(t, p, average='macro'), 4)


def AUCPR(t, p):
    return round(average_precision_score(t, p, average='macro'), 4)


def CCC(t, p):
    astd = np.std(t)
    bstd = np.std(p)
    am = np.mean(t)
    bm = np.mean(p)
    cc = pearsonr(t, p)[0]
    o = (2 * cc * astd * bstd) / (pow(astd, 2) + pow(bstd, 2) + pow(am - bm, 2))
    return round(o, 4)


def SAGR(t, p):
    o = 0
    for i in range(len(t)):
        o += (t[i] > 0) == (p[i] > 0)
    o /= len(t)
    return round(o, 4)


def conf_mat(t, p):
    return confusion_matrix(t, p)


def get_classifier_predictions(model, paths, labels):
    labels_t = []
    labels_t_r = []
    labels_p = []
    labels_p_r = []
    count = 0
    for i, path in enumerate(paths):
        img = image.load_img(path, target_size=(96, 96))
        img = image.img_to_array(img) / 255
        img = np.array(img).reshape((1, 96, 96, 3))
        labels_t_r.append(labels[i])
        labels_t.append(np.argmax(labels[i]))
        p = model.predict(img)
        labels_p_r.append(p)
        labels_p.append(np.argmax(p))
        count += 1
        print('Done:', count, end='\r')
    print('Done:', count)
    labels_t_r = np.array(labels_t_r).reshape((len(labels_t), 8))
    labels_p_r = np.array(labels_p_r).reshape((len(labels_t), 8))
    return labels_t, labels_p, labels_t_r, labels_p_r


def get_regressor_predictions(model, paths, labels):
    valence_t = []
    valence_p = []
    arousal_t = []
    arousal_p = []
    count = 0
    for i, path in enumerate(paths):
        img = image.load_img(path, target_size=(96, 96))
        img = image.img_to_array(img) / 255
        img = np.array(img).reshape((1, 96, 96, 3))
        valence_t.append(labels[i][0])
        arousal_t.append(labels[i][1])
        p = model.predict(img)
        valence_p.append(p[0][0])
        arousal_p.append(p[0][1])
        count += 1
        print('Done:', count, end='\r')
    print('Done:', count)
    return valence_t, valence_p, arousal_t, arousal_p

def eval(c_path=None, r_path=None):
    if c_path is None and r_path is None:
        print('Please specify a model')
        return
    v_paths_r = np.load('validation_paths.npy')
    v_labels_r = np.load('validation_labels.npy')
    if c_path is not None:
        print('** CALCULATING **')
        model = load_model(c_path)
        v_paths, v_labels, _ = process_data(CLASSIFY, v_paths_r, v_labels_r)
        true_l, pred_l, true_r, pred_r = get_classifier_predictions(model, v_paths, v_labels)
        print('** RESULTS **')
        print('ACC'.ljust(20), ACC(true_l, pred_l))
        print('F1'.ljust(20), F1(true_l, pred_l))
        print('KAPPA'.ljust(20), KAPPA(true_l, pred_l))
        print('AUCPR'.ljust(20), AUCPR(true_r, pred_r))
        print('AUC'.ljust(20), AUC(true_r, pred_r))
    if v_path is not None:
        print('** CALCULATING **')
        model = load_model(r_path)
        v_paths, v_labels, _ = process_data(REGRESS, v_paths_r, v_labels_r)
        valence_t, valence_p, arousal_t, arousal_p = get_regressor_predictions(model, v_paths, v_labels)
        print('** RESULTS **')
        print(''.ljust(20), 'VALENCE'.ljust(20), 'AROUSAL')
        print('RMSE'.ljust(20), str(RMSE(valence_t, valence_p)).ljust(20), RMSE(arousal_t, arousal_p))
        print('CORR'.ljust(20), str(CORR(valence_t, valence_p)).ljust(20), CORR(arousal_t, arousal_p))
        print('SAGR'.ljust(20), str(SAGR(valence_t, valence_p)).ljust(20), SAGR(arousal_t, arousal_p))
        print('CCC'.ljust(20), str(CCC(valence_t, valence_p)).ljust(20), CCC(arousal_t, arousal_p))

if __name__ == '__main__':
    try:
        if len(sys.argv) == 3:
            if sys.argv[1] == '-c':
                eval(c_path=sys.argv[2])
            elif sys.argv[1] == '-r':
                eval(r_path=sys.argv[2])  
            else:
                raise Exception()          
        elif len(sys.argv) == 5:
            c_path=None
            r_path=None
            if sys.argv[1] == '-c':
                eval(c_path=sys.argv[2], r_path=sys.argv[4])
            elif sys.argv[1] == '-r':
                eval(r_path=sys.argv[2], c_path=sys.argv[4])
            else:
                raise Exception()
        else:
            raise Exception()
    except:
        print('Invalid input, must be -c [path] or -r [path] or both')