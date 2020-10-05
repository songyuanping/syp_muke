import tensorflow as tf
import numpy as np
import sklearn as skl
from sklearn import datasets, model_selection
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import experimental, losses, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    prediction = tf.math.top_k(output, maxk).indices
    print(prediction)
    prediction = tf.transpose(prediction, perm=[1, 0])
    target_ = tf.broadcast_to(target, prediction.shape)
    correct = tf.equal(prediction, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * 100. / batch_size)
        res.append(acc)

    return res


output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob: ', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())
acc = accuracy(output, target, topk=(1, 2, 3, 4, 5, 6))
print('top 1-6 acc: ', acc)
