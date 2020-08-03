import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255
    y = tf.cast(y, tf.int32)
    return x, y


batchSize = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('dataSets: ', x.shape, y.shape, x_val.shape, y_val.shape)

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(50000).batch(batchSize)
db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batchSize)

network = Sequential([layers.Dense(256, activation=tf.nn.relu, input_shape=[None, 28 * 28]),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)
                      ])
network.summary()
for epoch in range(100):
    optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.75)
    for step, (x, y) in enumerate(db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,[-1, 28 * 28])
            out = network(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
            # 正则化参数
            loss_regularization = []
            for p in network.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss = loss + 0.0001 * loss_regularization

        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

        if step % 10 == 0:
            print('step: ', step, ' loss: ', float(loss), 'regularization: ', float(loss_regularization))

        if step % 50 == 0:
            total, total_correct = 0., 0
            for step1, (x1, y1) in enumerate(db_val):
                x1 = tf.reshape(x1, [-1, 28 * 28])
                out = network(x1)
                pred = tf.argmax(out, axis=1)
                pred = tf.cast(pred, tf.int32)
                correct = tf.equal(pred, y1)
                total_correct += tf.reduce_sum(tf.cast(correct, tf.int32)).numpy()
                total += x1.shape[0]

            print('step: ', step, " acc: ", float(total_correct / total))
