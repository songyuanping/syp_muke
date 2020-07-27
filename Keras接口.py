import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

x = tf.constant([2., 1., 0.1])
layer = layers.Softmax(axis=-1)
print(layer(x))
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 28 * 28])
model.summary()
model.save('model.h5')
# keras.experimental.export_saved_model(model,'model-savedmodel')
print('export saved model.')
print('saved total model.')
del model
model = keras.models.load_model('model.h5')
# model=keras.experimental.load_from_saved_model('model-savedmodel')
model.summary()


class MyDense(layers.Layer):
    # init和call是两个必须实现的函数
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
        self.kernel = tf.Variable(tf.random.normal([inp_dim, outp_dim]), trainable=False)

    def call(self, inputs, training=None):
        out = inputs @ self.kernel
        out = tf.nn.relu(out)
        return out


net = MyDense(4, 3)
print(net.variables, net.trainable_variables)

# resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)
# resnet.summary()
# x = tf.random.normal([4, 224, 224, 3])
# out = resnet(x)
# print(out.shape)

global_average_layer = layers.GlobalAveragePooling2D()
x = tf.random.normal([4, 7, 7, 2048])
out = global_average_layer(x)
print(out.shape)


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
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

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
network.fit(db, epochs=10, validation_data=ds_val, validation_freq=2)
network.evaluate(ds_val)
sample = next(iter(ds_val))
x = sample[0]
y = sample[1]
pred = network.predict(x)
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)
