import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch: ', sample[0].shape, sample[1].shape)

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

for epoch in range(10):
    # 对每个batch进行遍历
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])
        # tape只会跟踪tf.Variable类型的数据
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h2 = h1 @ w2 + tf.broadcast_to(b2, [h1.shape[0], 128])
            out = h2 @ w3 + b3
            y_onehot = tf.one_hot(y, depth=10)
            # shape:[b,10]
            loss = tf.square(out - y_onehot)
            loss = tf.reduce_mean(loss)

        lr = 0.001
        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 防止梯度爆炸或者梯度消失
        grads,_=tf.clip_by_global_norm(grads,10)
        # print('grads: ',grads)

        # w1 = w1 - lr * grads[0]会报错
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step % 100 == 0:
            print('epoch: ', epoch, 'step: ', step, 'loss shape:', loss.shape, loss)
