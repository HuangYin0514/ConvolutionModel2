# -*- coding: utf-8 -*-
# @Time     : 2019/1/24 21:36
# @Author   : HuangYin
# @FileName : MyMethod.py
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


def compute_content_cost(a_C, a_G):
    """
     Computes the content cost

     Arguments:
     a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
     a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

     Returns:
     J_content -- scalar that you compute using equation 1 above.
     """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(a_C)
    a_G_unrolled = tf.transpose(a_G)
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.pow((a_G_unrolled - a_C_unrolled), 2))
    return J_content
