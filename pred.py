from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from keras import backend as K
from scipy.io import savemat
from skimage.io import imsave

from data import load_data
from net import unet

weights_path = './weights_128.h5'
train_images_path = './data/train/'
test_dir_path = './data/test/'

gpu = '0'


def predict(mean=30.0, std=50.0):
    # load and normalize data
    if mean == 0.0 and std == 1.0:
        imgs_train, _, _ = load_data(train_images_path)
        mean = np.mean(imgs_train)
        std = np.std(imgs_train)

    imgs_test, imgs_mask_test, names_test = load_data(test_images_path)
    original_imgs_test = imgs_test.astype(np.uint8)

    imgs_test -= mean
    imgs_test /= std

    # load model with weights
    model = unet()
    model.load_weights(weights_path)

    # make predictions
    imgs_mask_pred = model.predict(imgs_test, verbose=1)

    # save to mat file for further processing
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)

    matdict = {
        'pred': imgs_mask_pred,
        'image': original_imgs_test,
        'mask': imgs_mask_test,
        'name': names_test
    }
    savemat(os.path.join(predictions_path, 'predictions.mat'),  matdict)


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = '/gpu:' + gpu

    for root, dirs, files in os.walk(test_dir_path):
        for dir in dirs:
            print(dir)
            test_images_path = root + dir + '/'
            predictions_path = './predictions/' + dir + '/'

            with tf.device(device):
                predict()






