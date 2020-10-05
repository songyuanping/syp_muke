import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x, y), (x_test, y_test) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch: ', sample[0].shape, sample[1].shape)

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
w4 = tf.Variable(tf.random.truncated_normal([64, 32], stddev=0.1))
b4 = tf.Variable(tf.zeros([32]))
w5 = tf.Variable(tf.random.truncated_normal([32, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

for epoch in range(100):
    # 对每个batch进行遍历
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])
        # tape只会跟踪tf.Variable类型的数据
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h2 = h1 @ w2 + tf.broadcast_to(b2, [h1.shape[0], 128])
            h3 = h2 @ w3 + b3
            h4 = h3 @ w4 + b4
            out = h4 @ w5 + b5
            y_onehot = tf.one_hot(y, depth=10)
            # shape:[b,10]
            loss = tf.square(out - y_onehot)
            loss = tf.reduce_mean(loss)

        lr = 0.005
        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5])
        # 防止梯度爆炸或者梯度消失
        grads, _ = tf.clip_by_global_norm(grads, 15)
        # print('grads: ',grads)

        # w1 = w1 - lr * grads[0]会报错
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        w4.assign_sub(lr * grads[6])
        b4.assign_sub(lr * grads[7])
        w5.assign_sub(lr * grads[8])
        b5.assign_sub(lr * grads[9])
        # if step % 100 == 0:
        #     print('epoch: ', epoch, 'step: ', step, 'loss shape:', loss.shape, loss)

    total_num, total_correct = 0, 0
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28 * 28])
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        h3 = tf.nn.relu(h2 @ w3 + b3)
        h4 = tf.nn.relu(h3 @ w4 + b4)
        # out.shape=[b,10]
        out = h4 @ w5 + b5

        prob = tf.nn.softmax(out, axis=1)
        # shape从[b,10]=>[b] 返回值为int64类型
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct * 100. / total_num
    print('acc: ', acc, '%')
