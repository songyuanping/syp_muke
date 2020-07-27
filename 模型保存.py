import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchSize = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets: ', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchSize)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).shuffle(10000).batch(batchSize)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential([layers.Dense(256, activation=tf.nn.relu),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)
                      ])
network.build(input_shape=[None, 28 * 28])
network.summary()


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = MyModel()
network.compile(optimizer=optimizers.Adam(lr=0.05),
                # 与tf.losses.categorical_crossentropy()不同
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db, epochs=1, validation_data=ds_val, validation_freq=1)
network.evaluate(ds_val)
network.save_weights('weights.ckpt')
print('saved weights!')
del network
network = MyModel()
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('weights.ckpt')
print('loaded weights!')
network.evaluate(ds_val)

network = Sequential([layers.Dense(256, activation=tf.nn.relu),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)
                      ])
network.build(input_shape=[None, 28 * 28])
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db,epochs=1,validation_data=ds_val,validation_freq=1)
network.evaluate(ds_val)
network.save('model.h5')
print('saved total model!')
del network
network = tf.keras.models.load_model('model.h5')
network.evaluate(ds_val)
