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
