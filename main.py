# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:24:19 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import sklearn as skl
from sklearn import datasets, model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import experimental, losses, layers, optimizers, Sequential, metrics

if __name__=='__main__':
    x = tf.constant([2., 1., 0.1])
    layer = layers.Softmax(axis=-1)
    print(layer(x))
    network = Sequential([
        layers.Dense(3, activation=None),
        layers.ReLU(),
        layers.Dense(2, activation=None),
        layers.ReLU()
    ])
    x = tf.random.normal([4, 3])
    print(network(x))
    layers_num = 2
    network = Sequential([])
    for i in range(layers_num):
        # 添加全连接层
        network.add(layers.Dense(3))
        # 添加激活函数层
        network.add(layers.ReLU())
    network.build(input_shape=(None, 4))
    print(network.summary())
    for p in network.trainable_variables:
        print(p.name, p.shape)

    network = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='relu'),
    ])
    network.build(input_shape=(None, 28 * 28))
    print(network.summary())

    # 设置测量指标为准确率
    network.compile(optimizer=optimizers.Adam(0.01), loss=losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)
    print(resnet.summary())
    x = tf.random.normal([4, 224, 224, 3])
    out = resnet(x)
    print('shape:', out.shape)
    print(out)
