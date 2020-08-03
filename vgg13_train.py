import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Sequential, metrics, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conv_layers = [
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),
    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),
    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),
    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),
    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),
]


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.squeeze(y)
    return x, y


batchSize = 128
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchSize)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchSize)
sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)


def main():
    convNet = Sequential(conv_layers)
    x = tf.random.normal([4, 32, 32, 3])
    out = convNet(x)
    print(out.shape)
    fc_net = Sequential([layers.Dense(256, activation=tf.nn.relu),
                         layers.Dense(128, activation=tf.nn.relu),
                         layers.Dense(100)
                         ])
    convNet.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    variables = convNet.trainable_variables + fc_net.trainable_variables
    optimizer = optimizers.Adam(lr=1e-4)
    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = convNet(x)
                out = tf.reshape(out, [-1, 512])
                out = fc_net(out)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True)
                # 计算每一个项的loss
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, 'step: ', step, 'loss: ', float(loss))

        if epoch % 2 == 1:
            total, total_correct = 0., 0
            for step, (x, y) in enumerate(test_db):
                out = convNet(x)
                out = tf.reshape(out, [-1, 512])
                out = fc_net(out)
                out = tf.nn.softmax(out)
                out = tf.argmax(out, axis=1)
                out = tf.cast(out, dtype=tf.int32)
                total += out.shape[0]
                correct = tf.equal(y, out)
                correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                total_correct += float(correct)
            print(epoch, 'acc: ', float(total_correct / total))


if __name__ == '__main__':
    main()
