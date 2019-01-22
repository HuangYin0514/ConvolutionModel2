# -*- coding: utf-8 -*-
# @Time     : 2019/1/19 21:44
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm


import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from week3.MyMethod import *

if __name__ == '__main__':
    sess = K.get_session()
    yolo_model = load_model("model_data/yolov2.h5")
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    out_scores, out_boxes, out_classes = predict(sess, "test.jpg",scores, boxes, classes, yolo_model,class_names)