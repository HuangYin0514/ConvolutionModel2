# -*- coding: utf-8 -*-
# @Time     : 2019/1/23 17:21
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from week4.FaceRecognition.MyMethod import *

if __name__ == '__main__':
    """
     my test----------------
     with tf.Session() as test:
        with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (
            tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
            tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
            tf.random_normal([3, 128], mean=3, stddev=4, seed=1),
        )
        loss = triplet_loss(y_true, y_pred)
        print("loss = ", str(loss.eval()))
    """

    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    # FRmodel.summary()
    print("Total Params = ", FRmodel.count_params())
    FRmodel.compile(optimizer="adam", loss=triplet_loss, metrics=['accuracy'])
    # load weights
    load_weights_from_FaceNet(FRmodel)

    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

    # It`s younes, welcome home!
    verify("images/camera_0.jpg", "younes", database, FRmodel)
    # It's not kian, please go away
    verify("images/camera_2.jpg", "kian", database, FRmodel)
    # it's younes, the distance is 0.6710074
    who_is_it("images/camera_0.jpg", database, FRmodel)