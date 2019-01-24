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


