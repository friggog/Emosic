import csv
import os
from math import sqrt

import cv2
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


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


def calc_ccc(a, b):
    astd = np.std(a)
    bstd = np.std(b)
    am = np.mean(a)
    bm = np.mean(b)
    p = pearsonr(a, b)[0]
    o = (2 * p * astd * bstd) / (pow(astd, 2) + pow(bstd, 2) + pow(am - bm, 2))
    return round(o, 4)


def cacl_sagr(a, b):
    o = 0
    for i in range(len(a)):
        o += (a[i] > 0) == (b[i] > 0)
    o /= len(a)
    return round(o, 4)


def calc_rmse(a, b):
    return round(mean_squared_error(a, b), 4)


def print_res(p, c):
    p1 = p  # [:, 0]
    c1 = c  # [:, 0]
    # p2 = p#[:, 1]
    # c2 = c#[:, 1]
    print('Metric'.ljust(20), 'Valence'.ljust(20), 'Arousal')
    print('RMSE'.ljust(20), str(calc_rmse(p1, c1)).ljust(20))  # , calc_rmse(p2,c2))
    print('CCC'.ljust(20), str(calc_ccc(p1, c1)).ljust(20))  # , calc_ccc(p2,c2))
    print('SAGR'.ljust(20), str(cacl_sagr(p1, c1)).ljust(20))  # , cacl_sagr(p2,c2))


def p_dist(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))


def extract_from_landmarks(landmarks):
    f = []
    # # eyebrow raise + chin
    # for i in range(0, 27):
    #     a = landmarks[i]
    #     f.append(p_dist(a, landmarks[33]))
    # # eye opening
    # for i in range(35, 42):
    #     a = landmarks[i]
    #     for j in range(35, 42):
    #         if i != j:
    #             b = landmarks[j]
    #             f.append(p_dist(a, b))
    # for i in range(42, 48):
    #     a = landmarks[i]
    #     for j in range(42, 48):
    #         if i != j:
    #             b = landmarks[j]
    #             f.append(p_dist(a, b))
    # # mouth
    # for i in range(48, 68):
    #     a = landmarks[i]
    #     for j in range(48, 68):
    #         if i != j:
    #             b = landmarks[j]
    #             f.append(p_dist(a, b))
    # ALL
    # c = np.array(landmarks).mean(axis=0)
    for a in landmarks:
        # f.append(p_dist(a, c))
        for b in landmarks:
            if a != b:
                f.append(p_dist(a, b))
    return f


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


def get_hogs(image, nx, ny, bins=8):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    h, w = image.shape[:2]
    sx, sy = int(w / nx), int(h / ny)
    fs = []
    for i in range(nx):
        for j in range(ny):
            m_hist = np.histogram(mag[j * sy:(j + 1) * sy, i * sx:(i + 1) * sx], bins=bins, density=True)[0]
            a_hist = np.histogram(angle[j * sy:(j + 1) * sy, i * sx:(i + 1) * sx], bins=bins, density=True)[0]
            fs.extend(m_hist)
            fs.extend(a_hist)
    return fs


def extract_hog_features(path, landmarks, c, s):
    # HOGS FOR ROIs
    (x, y) = c
    (w, h) = s
    path = '/Volumes/Charlie Hewitt\'s HDD/Affective/Manually_Annotated_Images/' + path
    image = cv2.imread(path)  # cut out face
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = []
    # # EYES
    # f.extend(get_hogs(make_box(landmarks[36:42], grey, b=2), nx=2, ny=2))
    # f.extend(get_hogs(make_box(landmarks[42:48], grey, b=2), nx=2, ny=2))
    # # EYEBROWS
    # f.extend(get_hogs(make_box(landmarks[17:22], grey), nx=2, ny=2))
    # f.extend(get_hogs(make_box(landmarks[22:27], grey), nx=2, ny=2))
    # # MOUTH
    # f.extend(get_hogs(make_box(landmarks[48:68], grey, b=2), nx=2, ny=2))
    face = grey[y:y + h, x:x + w]
    cv2.resize(face, (256, 256))
    f.extend(get_hogs(face, nx=16, ny=16, bins=8))
    return f


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
                # landmarks = []
                # abs_landmarks = []
                # cs = row[5].split(';')
                # for j in range(0, len(cs), 2):
                #     # normalise position using bounding box
                #     abs_landmarks.append((int(float(cs[j])), int(float(cs[j + 1]))))
                #     landmarks.append([(float(cs[j]) - x) / w, (float(cs[j + 1]) - y) / h])
                # labelset.append((valence, arousal))
                # fs = []
                # fs.extend(extract_from_landmarks(landmarks))
                # # fs.append(mouth_int(row[0], abs_landmarks))
                # fs.extend(extract_hog_features(row[0], abs_landmarks, (int(x),int(y)), (int(w),int(h))))
                # featureset.append(fs)
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

# clf = SVR(kernel='rbf', C=1000, gamma=0.01)
#
# clf.fit(train_f, np.array(train_l)[:,0])
# print_res(clf.predict(val_f), np.array(val_l)[:,0])
#
# clf.fit(train_f, np.array(train_l)[:,1])
# print_res(clf.predict(val_f), np.array(val_l)[:,1])

# clf = SVC(kernel='rbf', C=1000, gamma=0.01, class_weight='balanced')
# clf.fit(train_f, train_l)
# print(clf.score(val_f, val_l))

# print('TRAINING')
# reg = MultiOutputRegressor(GradientBoostingRegressor()).fit(train_f, train_l)
# print('VALIDATING')
# print_res(reg.predict(val_f), np.array(val_l))
#
# print('TRAINING')
# reg = MultiOutputRegressor(SVR()).fit(train_f, train_l)
# print('VALIDATING')
# print_res(reg.predict(val_f), np.array(val_l))
#
# print('TRAINING')
# reg = DecisionTreeRegressor().fit(train_f, train_l)
# print('VALIDATING')
# print_res(reg.predict(val_f), np.array(val_l))

# print('TRAINING')
# reg = RandomForestRegressor().fit(train_f, train_l)
# print('VALIDATING')
# print_res(reg.predict(val_f), np.array(val_l))
