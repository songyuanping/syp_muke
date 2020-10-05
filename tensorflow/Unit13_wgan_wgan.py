import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # z:[b,100] => [b,3*3*512] => [b,3,3,512] => [b,64,64,3]
        self.fc = layers.Dense(3 * 3 * 512)
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        # self.conv1 = layers.Conv2DTranspose(256, 3, 2, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        # self.conv2 = layers.Conv2DTranspose(128, 3,3, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        # [z,100] => [z,3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        # layers.Conv2DTranspose(256, 3, 3, 'valid') output:[b,9,9,256] (9-3)/3+1=3
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        # print("shape1: ",x.shape)
        # layers.Conv2DTranspose(128, 5, 2, 'valid') output:[b,21,21,128] (21-5)/2+1=9
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # print("shape2: ", x.shape)
        # self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid') output:[b,64,64,3] (64-4)/3+1=21
        x = self.conv3(x)
        # print("shape3: ", x.shape)
        x = tf.tanh(x)
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3] => [b,1]
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training))
        # [b,h,w,c] => [b,-1]
        x = self.flatten(x)
        # [b,-1] => [b,1]
        logits = self.fc(x)
        return logits


def main():
    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])
    prob = d(x)
    print(prob)
    x_hat = g(z)
    # output: [b, 64, 64, 3]
    print(x_hat.shape)


if __name__ == '__main__':
    main()
