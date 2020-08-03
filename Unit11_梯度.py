import tensorflow as tf
import numpy as np
import sklearn as skl
from sklearn import datasets, model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import experimental, losses, layers, optimizers, Sequential, metrics

W = tf.ones([2, 2])
eigenvalues = tf.linalg.eigh(W)[0]
print(eigenvalues)
val = [W]
for i in range(10):
    val.append([val[-1] @ W])
# 计算L2范数

norm = list(map(lambda x: tf.norm(x).numpy(), val))
print(val, norm)
W = tf.ones([2, 2]) * 0.4
eigenvalues = tf.linalg.eigh(W)[0]
print(eigenvalues)
val = [W]
for i in range(10):
    val.append(val[-1] @ W)
norm = list(map(lambda x: tf.norm(x).numpy(), val))
plt.plot(range(1, 12), norm)
# plt.show()
a = tf.random.uniform([2, 2])
a = tf.clip_by_value(a, 0.4, 0.6)
print(a)
a = tf.random.uniform([2, 2]) * 5
b = tf.clip_by_norm(a, 5)
print(a, b)
print(tf.norm(a), tf.norm(b))
a = tf.constant(tf.range(-6, 6))
x = tf.random.normal([2, 80, 100])
xt = x[:, 0, :]
cell = layers.LSTMCell(64)
# 初始化状态和输出List，[h,c]
state = [tf.zeros([2, 64]), tf.zeros([2, 64])]
out, state = cell(xt, state)
print(id(out), id(state[0]), id(state[1]))
for xt in tf.unstack(x, axis=1):
    out, state = cell(xt, state)
print(out, state)
layer = Sequential([layers.LSTM(64, return_sequences=True),
                    layers.LSTM(64, return_sequences=True),
                    layers.LSTM(64)])
out = layer(x)
print(out)

h = [tf.zeros([2, 64])]
cell = layers.GRUCell(64)
for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h)
print(out.shape)
net = Sequential([layers.GRU(64, return_sequences=True),
                  layers.GRU(64, return_sequences=True),
                  layers.GRU(64)])
out = net(x)
print(out.shape)
x=tf.random.normal([2,4,4,3],mean=1,stddev=0.5)
net=layers.BatchNormalization(axis=-1)
out=net(x)
print(net.variables)
out=net(x,training=True)
print(net.variables)
for i in range(100):
    out=net(x,training=True)
print(net.variables)