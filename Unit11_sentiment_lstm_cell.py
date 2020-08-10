import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers, datasets, losses, optimizers, Sequential

batchSize = 250
total_words = 100000
max_review_len = 200
embedding_len = 100

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
print('x_train shape:',x_train.shape,' x_test shape:',x_test.shape)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
print('x_train shape:',x_train.shape,' x_test shape:',x_test.shape)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5 * batchSize).batch(batchSize,
                                                                                               drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize, drop_remainder=True)


class MyLSTM(keras.Model):
    def __init__(self, units):
        super(MyLSTM, self).__init__()
        # 初始化[c0,h0]
        self.state0 = [tf.zeros([batchSize, units]), tf.zeros([batchSize, units])]
        self.state1 = [tf.zeros([batchSize, units]), tf.zeros([batchSize, units])]
        # [b,200] => [b,200,100]
        self.embedding = layers.Embedding(input_dim=total_words, input_length=max_review_len, output_dim=embedding_len)
        self.lstm_cell0 = layers.LSTMCell(units=units, dropout=0.5)
        self.lstm_cell1 = layers.LSTMCell(units=units, dropout=0.5)
        self.outlayer1 = layers.Dense(32)
        self.outlayer2 = layers.Dense(1)

    def call(self, inputs, training=None):
        '''
        model(x) 或者 model(x,training=True)时为训练模式
        model(x,training=False)时为测试模式
        :param inputs:
        :param training:
        :return:
        '''
        # out:[b,200,100]
        out = self.embedding(inputs)
        # [b,100]
        state0 = self.state0
        state1 = self.state1
        # word:[b,100]
        out1 = 0
        # 在时间的维度上进行展开,一次循环完成一个layer的工作，完成一次前向运算
        # 这里堆叠了两层，使得其表达能力更强
        for word in tf.unstack(out, axis=1):
            # x0，h0
            out0, state0 = self.lstm_cell0(word, state0, training)
            out1, state1 = self.lstm_cell1(out0, state1, training)
        # out1: [b,64] 不能使用relu激活函数
        prob = self.outlayer1(out1)
        prob = self.outlayer2(prob)
        # prob:[b,64] => [b,1]
        prob = tf.sigmoid(prob)
        return prob


def main():
    epochs = 5
    units = 128
    t0 = time.time()
    model = MyLSTM(units=units)
    model.compile(optimizer=optimizers.Adam(0.0005),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    model.fit(train_db, epochs=epochs, validation_data=test_db)
    # model.evaluate(test_db)
    t1 = time.time()
    print('total time cost:', t1 - t0)


if __name__ == '__main__':
    main()
