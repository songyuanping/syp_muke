import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# def mnist_dataset():
(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
# y = tf.one_hot(y, depth=10)
# y_val = tf.one_hot(y_val, depth=10)

batchSize = 256
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.map(prepare_mnist_features_and_labels)
ds = ds.shuffle(60000).batch(batchSize)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(prepare_mnist_features_and_labels)
ds_val = ds_val.shuffle(10000).batch(batchSize)
ds_iter = iter(ds)
sample = next(ds_iter)
print('batch: ', sample[0].shape, sample[1].shape)

model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dropout(0.25),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dropout(0.25),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dropout(0.25),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 28 * 28])
model.summary()
# w=w-lr*grad


# return ds, ds_val

def main():
    for epoch in range(60):
        optimizer = optimizers.Adam(lr=0.002*(1-tf.cast(epoch,tf.float32)/60.))
        for step, (x, y) in enumerate(ds):
            # x:[b,28,28] => [b,784]
            # y:[b]
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                # [b,784]=>[b,10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss_mse, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print('step:', step, 'loss: ', float(loss_ce), float(loss_mse))

        # test
        total_correct = 0
        total_num = 0
        for x, y in ds_val:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            # logits=>prob [b,10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b,10]=>[b]
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.reduce_sum(tf.cast(correct, tf.int32))
            total_correct += correct
            total_num += x.shape[0]

        acc = float(total_correct / total_num)
        print('epoch: ', epoch, 'acc: ', acc)


if __name__ == '__main__':
    main()
