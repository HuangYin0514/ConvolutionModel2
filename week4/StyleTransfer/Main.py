# -*- coding: utf-8 -*-
# @Time     : 2019/1/24 21:35
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
from week4.StyleTransfer.MyMethod import *

if __name__ == '__main__':
    # load model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    print(model)

    # look image
    content_image = scipy.misc.imread("images/louvre.jpg")
    imshow(content_image)
    plt.show()

    """
    # test compute_content_cost
    tf.reset_default_graph()
    with tf.Session() as test:
        tf.set_random_seed(1)
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        print("J_content = " + str(J_content.eval()))
        
    """

