import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def build_graph(input_shape, num_classes):
    """构建CNN模型"""
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input')
    labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    flatten = tf.layers.flatten(pool2, name='flatten')
    dense = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu, name='dense')
    logits = tf.layers.dense(inputs=dense, units=num_classes, name='logits')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name='loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss, name='optimizer')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float32),
                              name='accuracy')

    return inputs, labels, optimizer, loss, accuracy


def train_and_save_model(epochs=10, batch_size=128, save_path='model_1'):
    """训练并保存CNN模型"""
    # 加载MNIST数据集
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 构建图并创建会话
    input_shape = [None, 28, 28, 1]
    num_classes = 10
    inputs, labels, optimizer, loss, accuracy = build_graph(input_shape, num_classes)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 训练模型
    num_batches = mnist.train.num_examples // batch_size
    for epoch in range(epochs):
        for batch in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([-1, 28, 28, 1])
            sess.run(optimizer, feed_dict={inputs: batch_xs, labels: batch_ys})
            if batch % 100 == 0:
                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={inputs: batch_xs, labels: batch_ys})
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch}/{num_batches}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')

    # 保存模型
    graph_def = sess.graph.as_graph_def()
    tf.train.write_graph(graph_def, save_path, 'model.graphdef', as_text=False)
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    train_and_save_model()
# todo 这个是一个tf1.x更加完善的模型，但是发布推理遇到了问题是output name必须是loss,以后完善
