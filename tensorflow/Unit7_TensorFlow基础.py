import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

if __name__ == '__main__':
    x = tf.constant(1.2)
    x1 = tf.constant([1, 2, 1.2])
    print(x.ndim, x1.ndim)
    a = np.arange(5)
    aa = tf.convert_to_tensor(a)
    print(a.dtype, aa)
    index = tf.range(10)
    # 从右往左取数据
    print(index[-1:0:-1])
    index = tf.random.shuffle(index)
    a = tf.random.normal([10, 784])
    b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
    # 将a和b按照index中的下标进行打乱
    a = tf.gather(a, index)
    b = tf.gather(b, index)
    out = tf.random.uniform([4, 10])
    y = tf.range(4)
    print(y)
    y = np.arange(4)
    print(y)
    y = tf.one_hot(y, depth=10)
    loss = tf.keras.losses.mse(y, out)
    loss = tf.reduce_mean(loss)
    print(loss)
    x = tf.random.normal([4, 28, 28, 3])
    print(x[0, ..., 1, :].shape)
    a = tf.random.normal([4, 28, 28, 3])
    a = tf.reshape(a, [4, -1])
    print(a.shape)
    a = tf.random.normal([4, 3, 2, 1])
    # tranpose函数让矩阵转置
    print(tf.transpose(a).shape, tf.transpose(a, perm=[0, 1, 3, 2]).shape)
    a = tf.zeros([1, 2, 1, 3])
    print(tf.squeeze(a).shape)
    print(tf.squeeze(a, axis=-2).shape, tf.squeeze(a, axis=0).shape)
    a = tf.constant([[0], [10], [20], [30]])
    b = tf.constant([0, 1, 2])
    c = a + b
    print(c)
    a = tf.ones([4, 3, 2])
    b = [[1.1], [1.3], [0]]
    c = a + b
    print(c)
    a = tf.random.normal([1, 4, 1, 1])
    b = tf.random.normal([4, 32, 32, 3])
    a = tf.tile(a, [4, 8, 32, 3])
    print(a.shape)
    c = a + b
    print(c.shape)
    print([tf.math.log(10.) / tf.math.log(2.)])
    print([tf.math.log(3.) / tf.math.log(2.)])
    print(tf.exp(3.))
    a = tf.random.normal([2, 3, 4])
    b = tf.random.uniform([2, 4, 5])
    c = a @ b
    print(c)
    a = tf.ones([4, 35, 8])
    b = tf.ones([2, 35, 8])
    c = tf.concat([a, b], axis=0)
    print(c.shape)
    a = tf.ones([4, 32, 8])
    b = tf.ones([4, 3, 8])
    # concat不产生新的维度除合并的维度外都要一样
    c = tf.concat([a, b], axis=1)
    print(c.shape)
    # stack会产生新的维度，原来的维度都要一样
    a = tf.ones([4, 35, 8])
    b = tf.ones([4, 35, 8])
    c = tf.stack([a, b], axis=0)
    print(c.shape)
    # unsatck将该维度上的数据拆分为1
    c = tf.unstack(a, axis=2)
    print(c[0].shape, c[7].shape)
    res = tf.split(b, axis=2, num_or_size_splits=[2, 2, 4])
    print(res[0].shape, res[2].shape)
    a = tf.ones([2, 3])
    print(tf.norm(a), tf.sqrt(tf.reduce_sum(tf.square(a))))
    a = tf.ones([3, 4])
    # axis = 0 指在每一列上的元素相加，[m,n]计算后结果为[1,n],axis=1指在每一行上的元素相加,[m,n]计算后结果为[1,m]
    print(a, tf.norm(a).numpy(), tf.norm(a, ord=1, axis=0).numpy(), tf.norm(a, ord=1, axis=1).numpy())
    print(tf.norm(a, ord=2, axis=0).numpy(), tf.norm(a, ord=2, axis=1).numpy())
    a = tf.random.normal([4, 10])
    # 每10个当中求解一个最值
    print(tf.reduce_min(a, axis=1), tf.reduce_max(a, axis=1), tf.reduce_mean(a, axis=1))
    print(a)
    print(tf.argmin(a, axis=1), tf.argmax(a), tf.argsort(a))
    a = [4, 2, 3, 2, 1, 0, 1]
    a1, index = tf.unique(a)
    print(a1)
    a = tf.gather(a1, index)
    print(a)
    a = tf.random.shuffle(tf.range(10))
    a1 = tf.sort(a, direction='DESCENDING')
    a2 = tf.argsort(a, direction='DESCENDING')
    a = tf.gather(a, a2)
    print(a)
    print(a1, a2)
    a3 = tf.sort(a, direction='ASCENDING')
    print(a3)
    a = tf.random.uniform([4, 5], maxval=10, dtype=tf.int32)
    a = tf.sort(a)
    print(a)
    a = tf.sort(a, direction='DESCENDING')
    print(a)
    index = tf.argsort(a, direction='ASCENDING')
    print(index)
    a = tf.random.uniform([4, 5], maxval=10, dtype=tf.int32)
    res = tf.math.top_k(a, 3)
    print(a, res.indices, res.values)
    a = tf.reshape(tf.range(9), [3, 3])
    a = tf.pad(a, [[1, 1], [1, 1]])
    print(a)
    a = tf.random.normal([4, 28, 28, 3])
    a = tf.pad(a, [[0, 0], [2, 2], [2, 2], [0, 0]])
    print(a.shape)
    a = tf.range(10)
    a = tf.maximum(a, 2)
    print(tf.maximum(a, 2))
    a = tf.minimum(a, 8)
    print(tf.minimum(a, 8))
    print(tf.clip_by_value(a, 2, 8))
    a = tf.random.normal([3, 5], mean=5)
    print(tf.norm(a))
    a1 = tf.clip_by_norm(a, 4)
    print(a1, tf.norm(a1))
    a = tf.random.normal([4, 4])
    mask = a > 0
    print(mask)
    a1 = tf.boolean_mask(a, mask)
    print(a1)
    index = tf.where(mask)
    a1 = tf.gather_nd(a, index)
    print(a1)
    a = tf.ones([4, 4])
    b = tf.zeros([4, 4])
    c = tf.where(mask, a, b)
    print(c)
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    # 只能在元素全为0的Tensor上进行更新
    a = tf.scatter_nd(indices, updates, shape)
    print(a)
    y = tf.linspace(-3., 3, 1000)
    x = tf.linspace(-3., 3, 1000)
    points_x, points_y = tf.meshgrid(x, y)
    print(points_x.shape, points_y.shape)
    points = tf.stack([points_x, points_y], axis=2)
    print('points: ', points.shape)

    # z = tf.math.sin(points[..., 0]) + tf.math.cos(points[..., 1])
    # print('z: ', z.shape)
    # plt.figure('plot 2d func value')
    # plt.imshow(z, origin='lower', interpolation='none')
    # plt.colorbar()
    #
    # plt.figure('plot 2d func contour')
    # plt.contour(points_x, points_y, z)
    # plt.colorbar()
    # plt.show()

    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x.shape, y.shape, x.min(), x.max(), x.mean())
    print(x_test.shape, y_test.shape)
    print(y[:4])
    y_onehot = tf.one_hot(y, depth=10)
    print(y_onehot[:4])
    a = tf.random.normal([10])
    print(a)
    # 将值限定于-1到1之间
    a = tf.tanh(a)
    print(a)
    y = tf.constant([1, 2, 3, 0, 2])
    y = tf.one_hot(y, depth=4)
    y = tf.cast(y, dtype=tf.float32)
    out = tf.random.normal([5, 4])
    loss1 = tf.reduce_mean(tf.square(y - out))
    # norm为差的平方之和再开方
    loss2 = tf.square(tf.norm(y - out)) / (5 * 4)
    # reduce_mean将差的平方之和再除以元素的总个数
    loss3 = tf.reduce_mean(tf.losses.MSE(y, out))
    # tf.losses.MSE函数返回的是[b]的向量
    print(tf.losses.MSE(y, out))
    # 三者相等
    print(loss1, loss2, loss3)
    a = tf.fill([10], 0.1)
    b = a * tf.math.log(a) * tf.math.log(2.)
    c = -tf.reduce_sum(a * tf.math.log(a) * tf.math.log(2.))
    print(b, c)
    a = tf.constant([0.1, 0.1, 0.1, 0.7])
    print(-tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))
    a = tf.constant([0.01, 0.01, 0.01, 0.97])
    print(-tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))
    a = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25])
    print(a)
    a = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.8, 0.])
    print(a)
    a = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01])
    print(a)

    print(tf.losses.BinaryCrossentropy()([1], [0.1]))
    print(tf.losses.binary_crossentropy([1], [0.1]))
    x = tf.random.normal([1, 784])
    w = tf.random.normal([784, 2])
    b = tf.zeros([2])
    logits = x @ w + b
    prob = tf.math.softmax(logits)
    print(logits, prob)
    # 推荐使用，此时得到的结果更加稳定
    print(tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True))
    print(tf.losses.categorical_crossentropy([0, 1], prob))

    w = tf.Variable(1.)
    b = tf.Variable(2.)
    x = tf.Variable(3.)

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            y = x * w + b
        dy_dw, dy_db = tape2.gradient(y, [w, b])
    dy2_dw2 = tape1.gradient(dy_dw, [w])
    print(dy_dw, dy_db, dy2_dw2)
    assert dy_dw.numpy() == 3.
    # assert dy2_dw2 is None
    a = tf.linspace(-10., 10., 10)
    with tf.GradientTape() as tape:
        tape.watch(a)
        y = tf.sigmoid(a)
    grads = tape.gradient(y, [a])
    print(a, '\n', y, '\n', grads)
    a = tf.linspace(-1., 1., 10)
    print(tf.nn.leaky_relu(a))

    x = tf.random.normal([2, 4])
    w = tf.random.normal([4, 3])
    b = tf.zeros([3])
    y = tf.constant([2, 0])
    with tf.GradientTape() as tape:
        tape.watch([w, b])
        prob = tf.nn.softmax(x @ w + b, axis=1)
        loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))
    grads = tape.gradient(loss, [w, b])
    print('grads[0]:', grads[0])
    print('grads[1]:', grads[1])

    x = tf.random.normal([2, 4])
    w = tf.random.normal([4, 3])
    b = tf.zeros([3])
    y = tf.constant([2, 0])
    with tf.GradientTape() as tape:
        tape.watch([w, b])
        logits = x @ w + b
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))
    grads = tape.gradient(loss, [w, b])
    print(grads[0])
    print(grads[1])

    x = tf.random.normal([1, 3])
    w = tf.ones([3, 1])
    b = tf.ones([1])
    y = tf.constant([1])
    with tf.GradientTape() as tape:
        tape.watch([w, b])
        prob = x @ w + b
        loss = tf.reduce_mean(tf.losses.MSE(y, prob))
    grads = tape.gradient(loss, [w, b])
    print(grads)

    x = tf.random.normal([2, 4])
    w = tf.random.normal([4, 3])
    b = tf.zeros([3])
    y = tf.constant([2, 0])
    with tf.GradientTape() as tape:
        tape.watch([w, b])
        prob = tf.nn.softmax(x @ w + b, axis=1)
        loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))
        print('loss: ', loss)
    grads = tape.gradient(loss, [w, b])
    print(grads)

    # 链式法则
    x = tf.constant(1.)
    w1 = tf.constant(2.)
    b1 = tf.constant(1.)
    w2 = tf.constant(2.)
    b2 = tf.constant(1.)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([w1, b1, w2, b2])
        y1 = x * w1 + b1
        y2 = y1 * w2 + b2
    dy2_dy1 = tape.gradient(y2, [y1])
    dy1_dw1 = tape.gradient(y1, [w1])
    dy2_dw1 = tape.gradient(y2, [w1])
    print(dy2_dy1[0] * dy1_dw1[0], dy2_dw1[0])
    print(dy2_dy1, dy1_dw1,dy2_dw1)
    x=tf.range(25)+1
    x=tf.reshape(x,[1,5,5,1])
    x=tf.cast(x,tf.float32)
    w=tf.constant([[-1,2,-3.],[4,-5,6],[-7,8,-9]])
    w=tf.expand_dims(w,axis=2)
    w=tf.expand_dims(w,axis=3)
    print(w.shape)
    out=tf.nn.conv2d(x,w,strides=2,padding='VALID')
    print(out)
    xx=tf.nn.conv2d_transpose(out,w,strides=2,padding='VALID',output_shape=[1,5,5,1])
    print(xx)

    x=tf.range(16)+1
    x=tf.reshape(x,[1,4,4,1])
    x=tf.cast(x,tf.float32)
    w=tf.constant([[-1,2,-3.],[4,-5,6],[-7,8,-9]])
    w=tf.expand_dims(w,axis=2)
    w=tf.expand_dims(w,axis=3)
    out=tf.nn.conv2d(x,w,strides=1,padding='VALID')
    print(out)
    xx=tf.nn.conv2d_transpose(out,w,strides=1,padding='VALID',output_shape=[1,4,4,1])
    xx=tf.squeeze(xx)
    layer=tf.keras.layers.Conv2DTranspose(1,kernel_size=3,strides=1,padding='VALID')
    x1=layer(out)
    print(xx)
    print(x1)
    x=tf.random.normal([2,28,28,4])
    pool=layers.MaxPool2D(2,strides=2)
    out=pool(x)

    print(out.shape)
    pool=layers.MaxPool2D(3,strides=2)
    out=pool(x)
    print(pool.trainable_variables)
    print(out.shape)
    out=tf.nn.max_pool2d(x,2,strides=2,padding='VALID')
    print(out.shape)
    layer=layers.UpSampling2D(size=2)
    out=layer(x)
    print(out.shape)
    cell=layers.SimpleRNNCell(3)
    cell.build(input_shape=[None,4])
    print(cell.trainable_variables)

    h0=[tf.zeros([4,64])]
    x=tf.random.normal([4,80,100])
    xt=x[:,0,:]
    print(xt.shape)
    cell=layers.SimpleRNNCell(64)
    out,h1=cell(xt,h0)
    print(out.shape,h1[0].shape)
    print(id(out),id(h1[0]))
    h=h0
    for xt in tf.unstack(x,axis=1):
        out,h=cell(xt,h)
    print(out)

    x=tf.random.normal([4,80,100])
    # 取第一个时间戳的输入
    xt=x[:,0,:]
    cell0=layers.SimpleRNNCell(64)
    cell1=layers.SimpleRNNCell(64)
    h0=[tf.zeros([4,64])]
    h1=[tf.zeros([4,64])]
    for xt in tf.unstack(x,axis=1):
        out0,h0=cell0(xt,h0)
        out1,h1=cell1(out0,h1)
    print(cell0.trainable_variables)
    print(out1,h1)

    layer=layers.SimpleRNN(64)
    out=layer(x)
    print(out)
    layer=layers.SimpleRNN(64,return_sequences=True)
    out=layer(x)
    print(out.shape)
    # 除最末层外，都需要返回所有时间戳的输出
    net=Sequential([
        layers.SimpleRNN(64,return_sequences=True),
                    layers.SimpleRNN(64,return_sequences=True),
                    layers.SimpleRNN(64)])
    out=net(x)
    print(out)










