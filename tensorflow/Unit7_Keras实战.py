import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, losses, layers, optimizers, metrics, Sequential

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchSize = 125
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
# print('datasets: ', x.shape, y.shape, x_val.shape, y_val.shape)
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print('datasets: ', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(5*batchSize).batch(batchSize)
val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_db = val_db.map(preprocess).shuffle(5*batchSize).batch(batchSize)

sample = next(iter(train_db))
print('sample: ', sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        return inputs @ self.kernel + self.bias


class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 1024)
        self.fc2 = MyDense(1024, 512)
        self.fc3 = MyDense(512, 256)
        self.fc4 = MyDense(256, 128)
        self.fc5 = MyDense(128, 64)
        self.fc6 = MyDense(64, 32)
        self.fc7 = MyDense(32, 10)

    def call(self, inputs, training=None):
        # 对inputs进行reshape
        inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
        inputs = self.fc1(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc2(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc3(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc4(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc5(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc6(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.fc7(inputs)
        return inputs


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=100, validation_data=val_db, validation_freq=1)
# network.evaluate(val_db)
# network.save_weights('weights.ckpt')
# del network
# network = MyNetwork()
# network.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
#                 loss=losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# network.load_weights('weights.ckpt')
# print('load weights from file.')
# network.evaluate(val_db)
