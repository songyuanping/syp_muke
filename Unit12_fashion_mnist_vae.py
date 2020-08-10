import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers
from PIL import Image


def save_images(images, name):
    new_image = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            image = images[index]
            image = Image.fromarray(image, 'L')
            new_image.paste(image, (i, j))
            index += 1

    new_image.save(name)


h_dim = 20
z_dim = 10
batchSize = 500
lr = 1e-3

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(5*batchSize).batch(batchSize)
test_db = tf.data.Dataset.from_tensor_slices(x_test).batch(batchSize)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)

        # Decoder
        self.fc4 = layers.Dense(128, activation=tf.nn.relu)
        self.fc5 = layers.Dense(784)

    def encoder(self, inputs):
        h = self.fc1(inputs)
        # get mean
        mean = self.fc2(h)
        # get variance
        log_var = self.fc3(h)
        return mean, log_var

    def decoder(self, z):
        out = self.fc4(z)
        out = self.fc5(out)
        return out

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mean + std * eps
        return z

    def call(self, inputs, training=None):
        # [b,784] => [b,z_dim],[b,z_dim]
        mean, log_var = self.encoder(inputs)
        # reparameterize trick
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


model = VAE()
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam(lr=0.0015)

for epoch in range(1000):
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mean, log_var = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            # rec_loss=tf.reduce_mean(rec_loss)
            # 全局上的loss
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            # print(rec_loss)

            # compute KL divergence(mean,log_var) ~N(0,1)
            kl_div = -0.5 * (log_var + 1 - mean ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_mean(kl_div)
            loss = rec_loss + 1. * kl_div
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

    # evaluation
    z = tf.random.normal([batchSize, z_dim])
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/rec_epoch%d.png' % epoch)
