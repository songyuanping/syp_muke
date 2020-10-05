import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, datasets, layers, Sequential, metrics, optimizers


# 参数初始化有缺陷
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchSize = 200
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('dataSets: ', x.shape, y.shape, x_val.shape, y_val.shape)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(5*batchSize).batch(batchSize)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batchSize)

layer = Sequential([layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    # layers.Dropout(0.5),
                    layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    # layers.Dropout(0.5),
                    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    layers.Dense(10)
                    ])
layer.build(input_shape=[None, 28 * 28])
layer.compile(optimizer=optimizers.Adam(0.001),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
layer.fit(db, epochs=10, validation_data=db_val, validation_freq=1)
layer.evaluate(db_val)
