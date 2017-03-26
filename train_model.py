#!/usr/bin/python3
import vehiclerecognition as vr

import numpy as np
import cv2
import glob
import pickle
import os.path

from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sys import platform

DATA_FILE = 'data.p'

def load_images():
    car_features = []
    noncar_features = []

    for img_fn in glob.iglob('vehicles/**/*png'):
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_RGB2YCR_CB)
        car_features.append(vr.extract_features(img))

    for img_fn in glob.iglob('non-vehicles/**/*png'):
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_RGB2YCR_CB)
        noncar_features.append(vr.extract_features(img))

    car_features = np.stack(car_features)
    noncar_features = np.stack(noncar_features)

    size_dif = len(noncar_features) - len(car_features)
    if size_dif > 0:
        add_idx = np.random.randint(0, len(car_features)-1, size_dif)
        car_features = np.concatenate((car_features, car_features[add_idx]))
    elif size_dif < 0:
        add_idx = np.random.randint(0, len(noncar_features)-1, abs(size_dif))
        noncar_features = np.concatenate((noncar_features, noncar_features[add_idx]))

    x = np.append(car_features, noncar_features, axis=0)
    y = np.hstack((np.ones(car_features.shape[0]), np.zeros(noncar_features.shape[0])))
    return [x, y]


def load_data():
    print("About to load data")
    # Only try to save data if we don't run on Mac OS X because python 3.5 has a bug
    # on that platform when saving files larger than 2Gb.
    if platform != "darwin":
        if os.path.isfile(DATA_FILE):
            with open(DATA_FILE, "rb") as f:
                [x, y] = pickle.load(f)
        else:
            [x, y] = load_images()

            with open(DATA_FILE, "wb") as f:
                pickle.dump([x, y], f)
    else:
        [x, y] = load_images()

        print("shape of x %s %s"  % x.shape)
        print("shape of y %s"  % y.shape)

    print("Finished loading data")

    return [x, y]

def train_model():
    [x, y] = load_data()

    print(x.shape)

    pipe = Pipeline([
        ('Scaler', StandardScaler()),
        ('classify', LinearSVC())
    ])

    # Define the range over which the grid should search, currently at 5 can be increased.
    param_grid = {
        'classify__C': [0.1, 0.5, 1, 5],
        'classify__penalty': ['l1', 'l2'],
        'classify__loss': ['squared_hinge', 'hinge'],
         }

    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=2, n_jobs=2,
                        pre_dispatch='n_jobs',
                        scoring=metrics.make_scorer(metrics.scorer.roc_auc_score))
    clf = grid.fit(x, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    pickle.dump([clf, grid], open('classifier.p', 'wb'))


def main():
    print("About to train model")
    train_model()

if __name__ == "__main__":
    main()
