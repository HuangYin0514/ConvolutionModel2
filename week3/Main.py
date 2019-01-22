# -*- coding: utf-8 -*-
# @Time     : 2019/1/19 21:44
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm


from week3.yad2k.models.keras_yolo import yolo_head
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