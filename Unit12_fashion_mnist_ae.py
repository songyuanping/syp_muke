import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, experimental, losses, layers, optimizers, Sequential, metrics
from PIL import Image
from matplotlib import pyplot as plt


def save_images(images, name):
    new_image = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            image = images[index]
            image = Image.fromarray(image, mode='L')
            new_image.paste(image, (i, j))
            index += 1

    new_image.save(name)


h_dim = 20
batchSize = 512
lr = 0.02

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(5 * batchSize).batch(batchSize)
test_db = tf.data.Dataset.from_tensor_slices(x_test).batch(batchSize)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoder
        self.encoder = Sequential([layers.Dense(64, activation=tf.nn.relu),
                                   layers.Dense(256, activation=tf.nn.relu),
                                   layers.Dense(784)
                                   ])

    def call(self, inputs, training=None):
        # [b,784] => [b,h_dim]
        out = self.encoder(inputs)
        # [b,h_dim] => [b,784]
        out = self.encoder(out)
        return out


model = AE()
model.build(input_shape=(None, 784))
model.summary()

for epoch in range(100):
    optimizer = optimizers.Adam(lr=(1. / (epoch + 1)) * lr)
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print('epoch:', epoch, ' step: ', step, ' loss:', float(rec_loss))

        # evaluation
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b,784] => [b,28,28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        # [b,28,28] => [2b,28,28]
        # x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)
