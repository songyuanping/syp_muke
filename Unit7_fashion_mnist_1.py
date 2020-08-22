import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers, losses


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batch_size = 500
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
print('x_train:', x_train.shape, 'y_train:', y_train.shape)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.map(preprocess).shuffle(batch_size * 5).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batch_size)

model = Sequential([layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                    layers.Dense(10),
                    ])
model.build(input_shape=(None, 28 * 28))
model.summary()

for epoch in range(0, 20):
    optimizer = optimizers.Adam(learning_rate=0.0008)
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, [-1, 28 * 28])
            y_onehot = tf.one_hot(y, depth=10)
            logits = model(x)
            loss = tf.reduce_mean(losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 2 == 1:
        total_num, total_correct = 0., 0
        for step, (x, y) in enumerate(test_db):
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            total_num += x.shape[0]
            pred = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
            y = tf.cast(y, dtype=tf.int64)
            total_correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.int32))
        acc = float(total_correct) / total_num
        print('epoch:', epoch, 'acc:', float(acc), 'loss:', float(loss))
