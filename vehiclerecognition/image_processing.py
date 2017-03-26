import numpy as np
import cv2
from skimage.feature import hog

def color_hist(img, nbins=64, bins_range=(0,256)):
    Y_channel = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    Cr_channel = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    Cb_channel = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_feature = np.concatenate((Y_channel[0], Cr_channel[0], Cb_channel[0]))
    return hist_feature

def hog_features(img, orient=8, pix_per_cell=8, cell_per_block=2):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                   visualise=False)
    return features

def bin_spatial(img):
    Y = img[:,:,0].ravel()
    Cr = img[:,:,1].ravel()
    Cb = img[:,:,2].ravel()
    return np.hstack((Y, Cr, Cb))

def extract_features(img, size=64):
    img = cv2.resize(img, (size, size))
    hog_feat = np.concatenate((hog_features(img[:,:,0]),
                               hog_features(img[:,:,1]),
                               hog_features(img[:,:,2])))

    color_feat = color_hist(img)
    spatial_feat = bin_spatial(img)
    all_features = np.concatenate((hog_feat, color_feat, spatial_feat))
    return all_features
