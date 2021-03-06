# -*- coding: utf-8 -*-
# @Time     : 2019/1/20 14:33
# @Author   : HuangYin
# @FileName : MyMethod.py
# @Software : PyCharm

import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from keras import backend as K
from week3.yolo_utils import generate_colors, preprocess_image, draw_boxes, scale_boxes
from week3.yad2k.models.keras_yolo import yolo_boxes_to_corners


def yolo_filter_boxs(box_confience, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

        Arguments:
        box_confidence -- tensor of shape (19, 19, 5, 1)
        boxes -- tensor of shape (19, 19, 5, 4)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

        Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

        Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
        For example, the actual output size of scores would be (10,) if there are 10 boxes.
        """
    # compute the scores
    box_scores = np.multiply(box_confience, box_class_probs)
    #  Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_score = K.max(box_scores, axis=-1)
    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    #  same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = K.greater_equal(box_class_score, threshold)
    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_score, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

       Arguments:
       box1 -- first box, list object with coordinates (x1, y1, x2, y2)
       box2 -- second box, list object with coordinates (x1, y1, x2, y2)
       """
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
       Applies Non-max suppression (NMS) to set of boxes

       Arguments:
       scores -- tensor of shape (None,), output of yolo_filter_boxes()
       boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
       classes -- tensor of shape (None,), output of yolo_filter_boxes()
       max_boxes -- integer, maximum number of predicted boxes you'd like
       iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

       Returns:
       scores -- tensor of shape (, None), predicted score for each box
       boxes -- tensor of shape (4, None), predicted box coordinates
       classes -- tensor of shape (, None), predicted class for each box

       Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
       function will transpose the shapes of scores, boxes, classes. This is made for convenience.
       """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threashold=.6, iou_threashold=.5):
    """
        Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

        Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        box_xy: tensor of shape (None, 19, 19, 5, 2)
                        box_wh: tensor of shape (None, 19, 19, 5, 2)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

        Returns:
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box
        """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxs(box_confidence, boxes, box_class_probs, threshold=score_threashold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                      iou_threshold=iou_threashold)

    return scores, boxes, classes


def predict(sess, image_file, scores, boxes, classes, yolo_model,class_names):
    """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

        Arguments:
        sess -- your tensorflow/Keras session containing the YOLO graph
        image_file -- name of an image stored in the "images" folder.

        Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes

        Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
        """
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    print("Found {} boxes for {}".format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    plt.show()
    return out_scores, out_boxes, out_classes

