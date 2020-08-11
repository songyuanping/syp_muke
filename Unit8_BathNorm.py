import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

x = tf.random.normal([2, 4, 4, 3], mean=1, stddev=0.5)
layer = layers.BatchNormalization(axis=-1, center=True, scale=True, trainable=True)
out = layer(x)
print('forward in test mode: ', layer.variables)
out = layer(x, training=True)
print('forward in train mode (1 step): ', layer.variables)

for i in range(100):
    out = layer(x, training=True)
print('forward in train mode(100 step): ', layer.variables)

optimizer = optimizers.SGD(lr=0.01)
for i in range(10):
    with tf.GradientTape() as tape:
        out = layer(x, training=True)
        loss = tf.reduce_mean(tf.pow(out, 2)) - 1

    grads = tape.gradient(loss, layer.trainable_variables)
    optimizer.apply_gradients(zip(grads, layer.trainable_variables))
print('backward (10 steps)：', layer.variables)
