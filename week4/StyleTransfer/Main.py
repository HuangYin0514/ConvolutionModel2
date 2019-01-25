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
        
   
    # gram_matrix test
    tf.reset_default_graph()
    with tf.Session() as test:
        tf.set_random_seed(1)
        A = tf.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        print("GA = " + str(GA.eval()))
        
    
    # test compute_layer_style_cost
    tf.reset_default_graph()
    with tf.Session() as test:
        tf.set_random_seed(1)
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        print("GA = " + str(J_style_layer.eval()))
         """

    STYLE_LAYERS = [('conv1_1', 0.2),
                    ('conv2_1', 0.2),
                    ('conv3_1', 0.2),
                    ('conv4_1', 0.2),
                    ('conv5_1', 0.2)]
