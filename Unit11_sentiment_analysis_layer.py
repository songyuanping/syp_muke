import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, losses, optimizers, Sequential

batchSize = 250
total_words = 100000
max_review_len = 300
embedding_len = 100
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5 * batchSize).batch(batchSize,drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize,drop_remainder=True)

print('x_train shape:', x_train.shape, 'min:', float(tf.reduce_min(x_train)), 'max:', float(tf.reduce_max(x_train)))
print('x_test shape:', x_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b,300]=>[b,300,100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn = Sequential([layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
                               layers.SimpleRNN(units, dropout=0.5)])
        # [b,64]=>[b,1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        model(x) 或者 model(x,training=True)时为训练模式
        model(x,training=False)时为测试模式
        :param inputs:
        :param training:
        :return:
        """
        # embedding: [b, 300] => [b, 300, 100]
        out = self.embedding(inputs)
        # rnn cell compute
        # x: [b, 300, 100] => [b, 64]
        out = self.rnn(out, training)
        # x: [b, 64] => [b, 1]
        out = self.outlayer(out)
        return tf.sigmoid(out)


def main():
    units = 64
    epochs = 5
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_db, epochs=epochs, validation_data=test_db)
    model.evaluate(test_db)


if __name__ == '__main__':
    main()
