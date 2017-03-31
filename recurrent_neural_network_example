#coding:utf-8

"""
使用tensorflow做深度学习的步骤:
(1)首先定义计算图:Computational Graph

(2)其次是定义张量:Tensor

(3)创建session运行之,训练神经网络

(4)使用训练好的模型做预测  prediction
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("./datasets/data_Mnist/", one_hot=True)

# 迭代的次数
hm_epochs = 1

# 分类的数目
n_classes = 10

# 处理的每一批的容量
batch_size = 128

# 28x28像素的图片
chunk_size = 28
n_chunks = 28

# RNN循环结构的大小  越大越好  真的是循环结构的大小么
rnn_size = 256

# 定义两个输入tensor
# x是输入的值
x = tf.placeholder('float', [None, n_chunks,chunk_size])
# y是期望输出 即正确的类别
y = tf.placeholder('float')

print x, y

# 上面一直是在定义计算图
def recurrent_neural_network(x):
    # 这个多定义的一层是输出和最后的分类类别之间的那一层
    # 在我们自己的rnn_cell里面  已经有U,W,V这几个参数了
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    # placeholder之后为什么要转置一下了
    # ubuntu输入法翻页是- 和 =两个键
    # 对输入的转置  [1,0,2]这个是permute(转置)  意思是第0维转到第1维,第一维转到第一维,第二维转到第二维(不变)  例子如下:

    """
    
    """
    x = tf.transpose(x, [1, 0, 2])
    print x
    # 重塑输入x
    x = tf.reshape(x, [-1, chunk_size])
    print x
    # 分离输入x
    x = tf.split(0, n_chunks, x)
    print x
    # 为什么要做上面这几步?

    # 定义神经元的结构为LSTM的结构
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    # outputs是循环神经网络的输出,而states是循环神经网络的细胞状态
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    print outputs
    #
    # outputs[-1] 表示什么啊？  表示要将tensor展开成一个list
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


# 训练
def train_neural_network(x):
    # 预测的模型
    prediction = recurrent_neural_network(x)

    # 代价函数cost :交叉熵  cross_entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # 优化器  使用的是Adam优化器---梯度下降算法的一种优化方法
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # 创建session 并运行它
    with tf.Session() as sess:

        # 这一步必须得有  初始化全局变量
        sess.run(tf.global_variables_initializer())

        # 迭代的次数  一次迭代就是跑完一遍数据集
        for epoch in range(hm_epochs):
            # 每次迭代的损失
            epoch_loss = 0
            # 每次迭代中  要分批进行的  所以要在每一批里面做具体的逻辑操作
            for _ in range(int(mnist.train.num_examples / batch_size)):

                # 首先是每一次迭代中的x和y
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # x要reshape
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                # 给x和y喂数据  使用adam优化器和cost的损失函数  运行这个计算图网络
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # 将损失求和
                epoch_loss += c

            # 输出当前的迭代次数  和当前的损失
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # Tensorflow中的equal函数
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print(
        'Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)
