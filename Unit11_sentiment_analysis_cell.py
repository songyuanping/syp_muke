import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batchSize = 250
total_words = 90000
max_review_len = 300
embedding_len = 100
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, x_test.shape)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(batchSize * 5).batch(batchSize,
                                                                                               drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize, drop_remainder=True)
print('x_train shape：', x_train.shape, tf.reduce_min(y_train), tf.reduce_max(y_train))
print('x_test shape：', x_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batchSize, units])]
        self.state1 = [tf.zeros([batchSize, units])]

        # 将单词编码为长度为100的向量
        # [b,300] => [b,300,100]
        self.embedding = layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=max_review_len)
        # [b,300,100],h_dim:units
        # 在执行test模式时 dropout参数不起作用，使得test得到较好的效果
        self.rnn_cell0 = layers.SimpleRNNCell(units=units, dropout=0.5)
        self.rnn_dell1 = layers.SimpleRNNCell(units=units, dropout=0.5)

        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x),net(x,training=True): train mode
        net(x,training=False): test mode
        :param inputs: [b,300]
        :param training:
        :return:
        """
        # [b,300]
        x = inputs
        # [b,300] => [b,300,100]
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        out1 = 1
        # for word in x时，会在axis=0这个维度进行展开
        # 在时间的维度上进行展开
        for word in tf.unstack(x, axis=1):
            # h1=x*wxh+h0*whh
            # out0:[b,64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_dell1(out0, state1, training)
        # out: [b,64]=>[b,1]
        x = self.outlayer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)
        return prob


def main():
    units = 64
    epochs = 10

    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  # 前文已经进行了sigmoid函数，此处不需要再进行from_logits=True
                  loss=tf.losses.BinaryCrossentropy(),
                  # experimental_run_tf_function=False很重要
                  metrics=['accuracy'], experimental_run_tf_function=False
                  )
    model.fit(db_train, epochs=epochs, validation_data=db_test, validation_freq=2)
    # 执行test模式
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
