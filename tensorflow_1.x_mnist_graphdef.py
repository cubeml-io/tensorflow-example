import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def train(epochs, batch, model_dir):
    # 加载 MNIST 数据集
    mnist = input_data.read_data_sets('./../../dataset/MNIST/raw', one_hot=True)

    # 定义模型
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    y_true = tf.placeholder(tf.float32, [None, 10], name='labels')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='biases')
    y_pred = tf.nn.softmax(tf.matmul(x, W) + b, name='pred')

    # 定义损失函数和优化器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 创建会话并初始化变量
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 训练模型
    for i in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch)
        sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # 保存 GraphDef 模型
    graph_def = sess.graph.as_graph_def()
    tf.train.write_graph(graph_def, model_dir, 'model.graphdef', as_text=False)


if __name__ == '__main__':
    model_dir = "./../../model/tensorflow_mnist_tensorflow_mnist_graphdef"
    train(1000, 64, model_dir)
