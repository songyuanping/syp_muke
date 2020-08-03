import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, datasets, layers, Sequential, metrics, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchSize = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('dataSets: ', x.shape, y.shape, x_val.shape, y_val.shape)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(50000).batch(batchSize)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batchSize)

layer = Sequential([layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu6),
                    layers.Dropout(0.5),
                    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu6),
                    layers.Dropout(0.5),
                    layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu6),
                    layers.Dense(10)
                    ])
layer.build(input_shape=[None, 28 * 28])
layer.compile(optimizer=optimizers.Adam(0.01),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
layer.fit(db, epochs=10, validation_data=db_val, validation_freq=2)
layer.evaluate(db_val)
