import argparse
import json
import os
import struct
from array import array
import numpy as np
import tensorflow as tf

num_workers = 1


# 从 MNIST ubyte 文件中读取数据
class MnistDataloader(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.asarray(images), np.asarray(labels)

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            os.path.join(self.data_dir, 'train-images-idx3-ubyte'),
            os.path.join(self.data_dir, 'train-labels-idx1-ubyte')
        )
        x_test, y_test = self.read_images_labels(
            os.path.join(self.data_dir, 't10k-images-idx3-ubyte'),
            os.path.join(self.data_dir, 't10k-labels-idx1-ubyte')
        )
        return (x_train, y_train), (x_test, y_test)


def get_strategy(strategy='off'):
    strategy = strategy.lower()
    # 多机多卡
    if strategy == "multi_worker_mirrored":
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # 单机多卡
    if strategy == "mirrored":
        return tf.distribute.MirroredStrategy()
        # 单机单卡
    return tf.distribute.get_strategy()


def setup_env(args):
    tf.config.set_soft_device_placement(True)

    # limit the gpu memory usage as much as it need.
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Detected {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

        # OpenI-Octopus 多机多卡时设置 TF_CONFIG
    # 参考：
    # 1. https://github.com/tensorflow/docs/blob/r2.3/site/en/tutorials/distribute/multi_worker_with_keras.ipynb
    # 2. https://git.openi.org.cn/OpenI/octopus-doc/src/branch/master/docs/manual/train.md
    if args.strategy == 'multi_worker_mirrored':
        index = int(os.environ['VC_TASK_INDEX'])
        # task_name = os.environ["VC_TASK_NAME"].upper()
        ips = os.environ[f'VC_TASK_HOSTS']
        ips = ips.split(',')
        global num_workers
        num_workers = len(ips)
        ips = [f'{ip}:20000' for ip in ips]
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "worker": ips
            },
            "task": {"type": "worker", "index": index}
        })


def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default="./../../dataset/MNIST/raw")
    parser.add_argument('--output-dir', default="./../../model/tensorflow_mnist_savemodel")
    parser.add_argument('--epochs', default=5)
    parser.add_argument(
        '--strategy',
        default='off',
        choices=['off', 'mirrored', 'multi_worker_mirrored'],
        help='分别对应单机单卡、单机多卡、多机多卡'
    )
    args, unknown = parser.parse_known_args()
    return args


def mnist_dataset(batch_size=64, data_dir=None):
    # load dataset
    if data_dir:
        print(f'Loading mnist data from {data_dir}')
        (x_train, y_train), (x_test, y_test) = MnistDataloader(data_dir).load_data()
    else:
        print('Loading mnist data by tf.keras.datasets')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(
        batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset


def main():
    args = setup_config()
    # tf2 limitation: Collective ops must be configured at program startup
    strategy = get_strategy(args.strategy)
    setup_env(args)

    with strategy.scope():
        # build dataset
        train_dataset, test_dataset = mnist_dataset(
            batch_size=64 * num_workers,
            data_dir=args.data_dir
        )

        # build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        model.summary()

        # train and eval
    model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=70)
    model.evaluate(test_dataset, verbose=2)

    # save model
    if args.output_dir:
        # 为了验证多机分布式训练时，真实的跑在2个节点上，让在每个节点都生成模型文件
        # if 'VC_TASK_INDEX' in os.environ and int(os.environ['VC_TASK_INDEX']) > 0:
        #     return
        # current_time = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S-%f')
        # output_dir = os.path.join(
        #    args.output_dir,
        #    f"VC_TASK_INDEX_{os.environ['VC_TASK_INDEX']}_{current_time}"
        # )
        dir = args.output_dir + "/1/model.savedmodel"
        os.makedirs(dir, exist_ok=True)

        model.save(dir)
        print(f'Saved model to {os.listdir(dir)}')


if __name__ == '__main__':
    # 单机单卡 python tf2_dense_mnist_train.py
    # 单机多卡 python tf2_dense_mnist_train.py --strategy mirrored
    # 多机多卡 python tf2_dense_mnist_train.py --strategy multi_worker_mirrored
    print("TensorFlow version:", tf.__version__)
    main()
# mlflow.pytorch.save_model
