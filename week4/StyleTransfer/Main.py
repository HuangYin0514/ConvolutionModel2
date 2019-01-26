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
    """
    # load model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    print(model)

    # look image
    content_image = scipy.misc.imread("images/louvre.jpg")
    imshow(content_image)
    # plt.show()
        """

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

    """
    # test total_cost
    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J = " + str(J))
        """

    STYLE_LAYERS = [('conv1_1', 0.2),
                    ('conv2_1', 0.2),
                    ('conv3_1', 0.2),
                    ('conv4_1', 0.2),
                    ('conv5_1', 0.2)]

    # 初始化
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # 加载两张照片
    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)

    #对结果照片初始化
    generated_image = generate_noise_image(content_image)
    # imshow(generated_image[0])
    # plt.show()

    # 加载模型参数
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # 计算J
    sess.run(model["input"].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    sess.run(model["input"].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS, sess)

    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image, model, train_step, J, J_content, J_style)
