# -*- coding: utf-8 -*-
# @Time     : 2019/1/20 21:00
# @Author   : HuangYin
# @FileName : test.py
# @Software : PyCharm

import numpy as np
import tensorflow as tf

a = np.random.randn(3, 3, 3)
b = np.max(a, -1)
c = b > 0.5
print("a=" + str(a))
print("b=" + str(b))
print("c=" + str(c))
with tf.Session() as sess:
    d = tf.boolean_mask(a, c)
    print("d=" + str(d.eval(session=sess)))
