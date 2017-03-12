#!/usrs/bin/python3
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
import glob
import pickle

def color_hist(img, nbins=32, bins_range=(0,256)):
    Y_channel = np.histogram(img[:,:,0], bins=nbins, range=bins_range, density=True)
    Cr_channel = np.histogram(img[:,:,1], bins=nbins, range=bins_range, density=True)
    Cb_channel = np.histogram(img[:,:,2], bins=nbins, range=bins_range, density=True)
    hist_feature = np.concatenate((Y_channel[0], Cr_channel[0], Cb_channel[0]))
    return hist_feature

def hog_features(img, orient=8, pix_per_cell=8, cell_per_block=2):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                   visualise=False)
    return features


def extract_features(img, size=32):
    img = cv2.resize(img, (size, size))
    hog_feat = hog_features(img[:,:,0])
    color_feat = color_hist(img)
    all_features = np.concatenate((hog_feat, color_feat))
    return all_features


car_features = []
noncar_features = []

for img_fn in glob.iglob('vehicles/**/*png'):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2YCR_CB)
    car_features.append(extract_features(img))

for img_fn in glob.iglob('non-vehicles/**/*png'):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2YCR_CB)
    noncar_features.append(extract_features(img))

noncar_features = np.stack(noncar_features[:len(car_features)])
car_features = np.stack(car_features)

x = np.append(car_features[:N], noncar_features[:N], axis=0)
scaler = StandardScaler().fit(x)
X_scaled = scaler.transform(x)
y = np.hstack((np.ones(car_features[:N].shape[0]), np.zeros(noncar_features[:N].shape[0])))

from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, SparsePCA

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', SVC())
])

N_FEATURES_OPTIONS = [10, 50, int(X_scaled.shape[1] * .25), int(X_scaled.shape[1] * .5)]
print(N_FEATURES_OPTIONS)
# Define the range over which the grid should search, currently at 5 can be increased.
param_grid = {
    'reduce_dim': [PCA(iterated_power=7), SparsePCA()],
    'reduce_dim__n_components': N_FEATURES_OPTIONS,
    'classify__C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'classify__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=2, n_jobs=64,
                    pre_dispatch='2*n_jobs',
                    scoring=metrics.make_scorer(metrics.scorer.f1_score, average="binary"))
clf = grid.fit(X_scaled, y)

pickle.dump(clf, open('classifier.p', 'wb'))

