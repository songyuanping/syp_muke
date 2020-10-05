import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import experimental, losses, layers, optimizers, Sequential, metrics,models


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation(activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        # 下采样层（残差）
        if stride != 1:
            self.downsample = Sequential()
            # 使用1*1的核保证输出是一致的
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def call(self, inputs, training=None):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2, training=training)

        add = layers.add([residual, bn2])
        out = self.relu(add)
        return out


class ResNet(models.Model):

    def __init__(self, layers_dims, num_classes=100):
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation(tf.nn.relu),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='SAME')
                                ])
        self.layer1 = self.build_resblock(64, layers_dims[0])
        self.layer2 = self.build_resblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layers_dims[3], stride=2)

        # 将 [b,512,h,w]的后两个维度加和取均值，变成[b,512,1,1]的维度
        self.avgpool = layers.GlobalAvgPool2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # 输出[b,channel]
        x = self.avgpool(x)
        # 输出[b,num_classes]
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # 可能有下采样层，只采样一次
        res_blocks.add(BasicBlock(filter_num, stride))
        for i in range(blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
