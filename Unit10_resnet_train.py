import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers

from Unit10_ResNet import resnet18,resnet34


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255 - 0.5
    y = tf.cast(y, tf.int32)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # [b,32,32,3]=>[b,1,1,512]
    model = resnet18()
    # input_shape应该使用元组
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(500):

        for step, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x_train, training=True)
                y_onehot = tf.one_hot(y_train, depth=100)
                loss = losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))

        # if epoch % 10 == 9:
        total_num, total_correct = 0., 0
        for x, y in test_db:
            logits = model(x, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.int32))

            total_num += x.shape[0]
            total_correct += correct

        acc = tf.cast(total_correct,tf.float32) / total_num
        print('epoch:', epoch, "acc:", float(acc))


if __name__ == '__main__':
    main()
